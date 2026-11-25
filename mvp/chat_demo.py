"""MVP step 3: chat demo. ReSpeaker -> Silero VAD (0.8 s pause) -> Whisper -> OpenAI -> Piper.

Same microphone, VAD and Piper code as echo_demo.py; OpenAI answers instead of echoing.
--text-input: type instead of speaking (no microphone, no Whisper) to test the LLM part quickly.

Run (laptop):  export OPENAI_API_KEY=...   then
  ~/mvp_venv/bin/python chat_demo.py --text-input
  ~/mvp_venv/bin/python chat_demo.py --model base.en
Jetson: same environment variables as echo_demo.py, plus --model large-v3-turbo --device cuda --compute-type float16
Stop: Ctrl+C, or say/type "stop chat".
"""

import argparse
import csv
import json
import os
import time
import urllib.request
import subprocess
from pathlib import Path
import random
import threading
import collections
import struct
import usb.core                    # pyusb: read the ReSpeaker's own sound direction over USB
import http.server                 # built-in small web server: shows the live status page (no install)
import wave                        # save the sentence audio for the lip check
import re
from datetime import datetime
from difflib import SequenceMatcher
import echo_demo                     # same microphone, VAD and Piper code as echo_demo.py
from echo_demo import LOG_DIR, SAMPLE_RATE, record_sentence, speak, start_mic, start_piper, stop_mic

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")   # fast, non-reasoning chat model
HISTORY_TURNS = 5                                            # user + robot pairs sent as context
SYSTEM_PROMPT = (
    "You are a friendly social robot at FAU Erlangen. Several people may stand around you. "
    "You get one person's words from speech recognition, which may contain errors. "
    "Answer in 1-2 short spoken sentences, no lists, no emojis, no markdown. "
    # persona: most questions to public robots are about the robot itself (Raza et al. 2022)
    "People often ask about you: your name, age, who built you, how you feel, what you like, what you can do. "
    "Answer these in a friendly, playful way. You can talk, listen and see faces, but you cannot move, "
    "shake hands or dance: say so kindly. "
    "You can answer general knowledge questions and simple math. You have no internet, so you cannot check "
    "live facts like the weather or today's news. "
    # how to read behaviour (Vertegaal 2001, Katzenmaier et al. 2004): background knowledge, no fixed rules
    "A message may end with [camera: ...]: what your camera measured while the person spoke. "
    "Background on people: someone talking to you usually looks at you; someone talking to another person usually "
    "looks at that person. But people sometimes look at you while talking to a friend, and sometimes ask you "
    "something while looking elsewhere. Talking ABOUT you ('it', 'this robot') or about plans with others is "
    "usually meant for someone else. Use the words and the camera facts together to decide. "
    "If the person is not talking to you, answer exactly: [IGNORE]. "
    "If the text makes no sense, answer exactly: [REPEAT]. "
    "If the person says goodbye or wants to stop, say a short goodbye and end with [END]. "
    "The [camera: ...] part is hidden sensor data for you only: never mention it and never say what you saw the person do. "
    "A request asking YOU to do something, even for someone else ('say hello to my friend'), is meant for you. "
    "You are the social robot of the FAPS lab at FAU Erlangen and have no personal name. "
    "The person may be called Nila. Words speech recognition often gets wrong: Nila, FAU, Erlangen, FAPS, robot. "
    "You are friendly, playful and polite. You do not swear or insult anyone. ")

REPEAT_TEXT = "Sorry, I didn't catch that. Could you say it again?"
ENGLISH_ONLY = "Sorry, I didn't understand that. Could you say it again in English?"

STOP_PHRASES = ("stop chat",)
FILLERS = ["Hmm.", "Okay.", "Let me think."]   # short instant reaction while Whisper + LLM work (Shiwa 2009: filler softens long waits)
GAZE_PYTHON = os.environ.get("GAZE_PYTHON", os.path.expanduser("~/step1_vision_venv/bin/python"))
# at the settings (next to GAZE_PYTHON): which vision script; multi-person = ../vision/people_gaze.py
GAZE_SCRIPT = os.environ.get("GAZE_SCRIPT", str(Path(__file__).resolve().parent / "gaze_trigger.py"))
# GREETINGS: the robot says one of these when someone looks at it (gaze trigger).
# A different one each time, so it does not sound like a machine (professor, 3 Nov).
GREETINGS = [
    "Hello! How can I help you?",
    "Hi there! Nice to see you.",
    "Hey! What can I do for you?",
    "Hello! Do you have a question for me?",
    "Hi! I'm listening.",
    "Good to see you! How can I help?",
    "Hello! What would you like to know?",
    "Hi! Feel free to ask me anything.",
    "Hey there! How are you today?",
    "Hello! I'm here if you need me.",
]
# UNLOCK RULES (engagement): when does the robot let its partner go?
# Idea: engagement stays alive through "connection events" (gaze, speech turns); if they stop
# for too long, the person has disengaged (Rich et al., HRI 2010, p.2 and p.4).
# 5 s and 30 s are our own starting values (paper: robot game max gap 9.9 s, human study 70 s): tune in tests.
FACE_GONE_S = 5.0                   # no face for 5 s -> the person left -> unlock
NOT_ENGAGED_S = 15.0                # no look AND no speech for 30 s -> ask STILL_THERE first, then unlock
STILL_THERE = "So... anything else you want to talk about?"   # Bohus & Horvitz 2014 p.1: "So..." then "Anything else I can do for you?"
GOODBYE = "Okay, I'll catch you later then."                  # Bohus & Horvitz 2014 p.1: "Well, I'll catch you later then."


def ask_llm(history):
    """Send the conversation to OpenAI and return the robot's answer."""
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history[-2 * HISTORY_TURNS:],
    }).encode()
    request = urllib.request.Request(OPENAI_URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["choices"][0]["message"]["content"].strip()


def is_stop(text):
    words = text.lower().strip(" .!?")
    return any(p in words or SequenceMatcher(None, p, words).ratio() > 0.8 for p in STOP_PHRASES)

# ENGAGEMENT: gaze_trigger.py runs the whole time. This thread reads its lines in the background
# and remembers WHEN we last saw a face and WHEN the person last looked at the robot.
# The chat loop later uses these times to decide: stay with the partner, or unlock.
last_face = 0.0                    # time of the last LOOK or AWAY (a face was there)
last_look = 0.0                    # time of the last LOOK (the face looked at the robot)
last_talk = 0.0                    # time of the last speech in the chat (person or robot)
look_run = 0                       # how many LOOK frames in a row (1-2 = noise, 3 = real look)
doa_log = collections.deque(maxlen=600)   # (time, ReSpeaker angle) for the last 60 s, 10 per second
face_log = collections.deque(maxlen=600)  # (time, face angle in the camera) for the last 60 s, from gaze_trigger.py

look_log = collections.deque(maxlen=600)  # (time, True = LOOK / False = AWAY or NOFACE), for "looking while speaking"
DOA_CENTRE_DEG = 182.8             # calibrated 11 Nov 2026: camera angle = (182.8 - ReSpeaker angle) * 0.77
DOA_SCALE = 0.77
MATCH_TOL_DEG = 15.0               # voice within 15 deg of the face = from that face (step2_voice_face_match.py)

WEB_PORT = 8080                                   # open http://localhost:8080 (later: Pepper tablet shows the same page)
WEB_DIR = Path(__file__).parent / "web"           # index.html (the page) + camera.jpg (latest camera image)
status = {"state": "WAITING", "looking": False, "face_ago": None,      # everything the page shows, updated by main()
          "step2": None, "voice_deg": None, "face_deg": None,
          "turns": [], "times": {}, "not_engaged": 0}

controls = {"typed": None, "mute": False}        # set by the buttons on the web page

class WebHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # /status -> status as JSON, /camera.jpg -> latest image, anything else -> the page
        try:
            if self.path == "/status":
                now = time.perf_counter()
                status["face_ago"] = round(now - last_face, 1) if last_face else None   # seconds since a face was seen
                status["look_ago"] = round(now - last_look, 1) if last_look else None   # seconds since a look

                body, kind = json.dumps(status).encode(), "application/json"
            elif self.path.startswith("/camera.jpg"):
                body, kind = (WEB_DIR / "camera.jpg").read_bytes(), "image/jpeg"
            else:
                body, kind = (WEB_DIR / "index.html").read_bytes(), "text/html"
        except FileNotFoundError:
            self.send_error(404)                  # e.g. no camera image yet
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        # buttons on the page send {"cmd": "...", "value": ...} to /cmd
        global NOT_ENGAGED_S
        n = int(self.headers.get("Content-Length", 0))
        msg = json.loads(self.rfile.read(n) or b"{}")
        cmd, value = msg.get("cmd"), msg.get("value")
        if cmd == "type":
            controls["typed"] = str(value).strip() or None       # typed sentence, used instead of speech
        elif cmd == "mute":
            controls["mute"] = not controls["mute"]              # robot answers only on screen
        elif cmd == "not_engaged":
            NOT_ENGAGED_S = float(value)                         # partner_gone() uses the new value
        elif cmd == "note":
            with open(LOG_DIR / "test_notes.csv", "a", newline="") as f:   # pass/fail notes while testing
                csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), value])
        status["mute"], status["not_engaged_s"] = controls["mute"], NOT_ENGAGED_S
        self.send_response(204)                                  # 204 = ok, nothing to send back
        self.end_headers()


    def log_message(self, *args):
        pass                                      # don't print every request in the terminal


def start_web():
    # runs the web server in the background, next to the chat
    server = http.server.ThreadingHTTPServer(("0.0.0.0", WEB_PORT), WebHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Web page: http://localhost:{WEB_PORT}", flush=True)

look_trigger = threading.Event()   # switched on when gaze_trigger.py prints TRIGGER (looked for 1 s)


def read_gaze_lines(gaze):
    global last_face, last_look, look_run
    for line in gaze.stdout:
        parts, now = line.split(), time.perf_counter()
        word = parts[0] if parts else ""                         # LOOK / AWAY / NOFACE / TRIGGER
        if word in ("LOOK", "AWAY"):
            last_face = now
            if len(parts) > 1:
                face_log.append((now, float(parts[1])))       # where the face is (deg), e.g. "LOOK -1.3"

        if word in ("LOOK", "AWAY", "NOFACE"):
            look_log.append((now, word == "LOOK"))             # every frame: looking at the robot or not

            look_run = look_run + 1 if word == "LOOK" else 0   # count LOOKs in a row, reset on anything else
        if look_run >= 3:              # 3 LOOKs in a row = real look (Cheng et al. 2026, p.4: smooth window 3)
            last_look = now

        if word == "TRIGGER":
            look_trigger.set()
        if word == "LIPSCORE":                                   # gaze_trigger's answer to ask_lips()
            lip_reply["pct"] = None if parts[1] == "none" else int(parts[1])
            lip_reply["done"].set()


def read_doa():
    """Background thread: ReSpeaker direction (DOAANGLE) every 0.1 s, same USB call as step2_voice_face_match.py."""
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)        # ReSpeaker Mic Array v2.0
    if dev is None:
        print("ReSpeaker not found on USB: no direction check", flush=True)
        return
    while True:
        raw = dev.ctrl_transfer(0xC0, 0, 0x80 | 0 | 0x40, 21, 8, 100000)   # group 21, offset 0 = DOAANGLE
        try:
            raw = dev.ctrl_transfer(0xC0, 0, 0x80 | 0 | 0x40, 21, 8, 100000)   # group 21, offset 0 = DOAANGLE
        except usb.core.USBError:
            time.sleep(0.5)                     # one failed USB read (cable hiccup): skip it, keep the thread alive
            continue

        doa_log.append((time.perf_counter(), struct.unpack("ii", bytes(raw))[0]))
        time.sleep(0.1)

def look_pct(t_start, t_end):        # how much did the person look at the robot while speaking?"

    # returns % of gaze frames during the sentence where the person looked at the robot (None = no frames)
    frames = [looking for t, looking in look_log if t_start <= t <= t_end]
    return round(100 * sum(frames) / len(frames)) if frames else None


def hint_line(look, lips):    # hint_line: "turn the numbers into one text line for the LLM".
    # measured facts for the LLM, added after the user's words (more facts later: people count, VLM, ...),It collects the facts we have: look % and lips %.
    facts = []
    if look is not None:
        facts.append(f"looked at you {look} % of the time while speaking")
    if lips is not None:
        facts.append(f"lips moving {lips} %")
    return f" [camera: {' | '.join(facts)}]" if facts else ""


def voice_from_partner(t_start, t_end):
    """Step 2 for one sentence: did most of the sound direction come from the partner's face?
    Returns (True / False / None = no data, voice deg, face deg). Majority over the whole sentence,
    because the direction needs about 1 s to settle (home dataset, 11 Nov 2026)."""
    doas = [d for t, d in doa_log if t_start <= t <= t_end]
    faces = [a for t, a in face_log if t_start <= t <= t_end]
    if not doas or not faces:
        return None, None, None
    voice = [((DOA_CENTRE_DEG - d + 180) % 360 - 180) * DOA_SCALE for d in doas]
    face = sorted(faces)[len(faces) // 2]                       # median face position
    close = sum(abs(v - face) <= MATCH_TOL_DEG for v in voice)
    return close >= len(voice) / 2, round(sorted(voice)[len(voice) // 2], 1), round(face, 1)

LIP_MIN_PCT = 50                   # lips "speaking" in at least half of the frames = the partner talks (offline test rule)
LIP_WAV = LOG_DIR / "lip_last.wav" # sentence audio for gaze_trigger's lip check
lip_reply = {"pct": None, "done": threading.Event()}   # filled by read_gaze_lines when gaze_trigger answers LIPSCORE


def ask_lips(gaze, audio, t_start, t_end):
    # save the sentence audio and ask gaze_trigger: "were the partner's lips speaking between t_start and t_end?"
    with wave.open(str(LIP_WAV), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes((audio.clip(-1, 1) * 32767).astype("<i2").tobytes())
    lip_reply["done"].clear()
    gaze.stdin.write(f"LIP {t_start:.3f} {t_end:.3f} {LIP_WAV}\n")
    gaze.stdin.flush()


def lip_result(timeout=5.0):
    # wait for gaze_trigger's answer (it works while Whisper works, so it is usually ready already)
    return lip_reply["pct"] if lip_reply["done"].wait(timeout) else None



def start_gaze(device):
    """Start gaze_trigger.py ONCE. It keeps the camera on for the whole demo (also during the chat),
    and the background thread reads its LOOK / AWAY / NOFACE / TRIGGER lines."""
    cmd = [GAZE_PYTHON, "-u", GAZE_SCRIPT] + (["--device", device] if "gaze_trigger" in GAZE_SCRIPT else ["0"])
    gaze = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    threading.Thread(target=read_gaze_lines, args=(gaze,), daemon=True).start()   # daemon: stops when the demo stops
    return gaze


def wait_for_look():
    """Wait until someone has looked at the camera for 1 s (gaze_trigger.py printed TRIGGER)."""
    print("Waiting: look at the camera for 1 s ...", flush=True)
    look_trigger.clear()   # forget old TRIGGERs (e.g. from looking during the last chat)
    look_trigger.wait()    # sleep here until read_gaze_lines switches it on

def partner_gone():
    """Asked by record_sentence about every 0.1 s while nobody speaks.
    Returns WHY to stop listening ("left" / "not engaged"), or None = keep waiting."""
    now = time.perf_counter()
    if now - last_face > FACE_GONE_S:
        return "left"                          # no face for 5 s
    if now - max(last_look, last_talk) > NOT_ENGAGED_S:
        return "not engaged"                   # no look and no speech for 30 s
    return None

def make_wav(text, wav_path):
    # Piper only writes the wav file (no playback): used once at start for the filler sounds
    echo_demo._piper.stdin.write(json.dumps({"text": text, "wav": str(wav_path)}) + "\n")
    echo_demo._piper.stdin.flush()
    echo_demo._piper.stdout.readline()          # "done"

def robot_says(text, wav_path):
    # speak, unless muted on the web page (then the answer is only shown on screen)
    if controls["mute"]:
        return 0.0, time.perf_counter()
    return speak(text, wav_path)

def main():
    global last_talk, NOT_ENGAGED_S        # main changes these shared values

    parser = argparse.ArgumentParser()
    parser.add_argument("--text-input", action="store_true", help="type instead of speaking")
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--lang-model", default=None, help="e.g. tiny: only detects the language, --model base.en writes the text")

    parser.add_argument("--gaze", action="store_true", help="start each conversation by looking at the camera")
    parser.add_argument("--gaze-device", default="cpu", help="cpu (laptop) or cuda (Jetson)")
    parser.add_argument("--not-engaged-s", type=float, default=NOT_ENGAGED_S,
                        help="no look and no speech this long -> 'So...', same again -> unlock")

    args = parser.parse_args()
    NOT_ENGAGED_S = args.not_engaged_s     # use the value from the command line (partner_gone() reads it)
    status["not_engaged_s"] = NOT_ENGAGED_S                # shown in the page's "Not engaged" box

    if not args.text_input:
        from faster_whisper import WhisperModel
        from faster_whisper.vad import get_vad_model
        print(f"Loading {args.model} on {args.device} ({args.compute_type}) ...", flush=True)
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        lang_model = WhisperModel(args.lang_model, device=args.device, compute_type=args.compute_type) if args.lang_model else None

        vad = get_vad_model()

    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source = "text" if args.text_input else args.model.replace("/", "_")
    log_path = LOG_DIR / f"chat_{source}_{args.device}_{stamp}.csv"
    wav_path = LOG_DIR / "piper_last.wav"
    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["turn", "user_text", "robot_text", "whisper_s", "llm_s", "piper_synthesis_s",
                  "speech_end_to_speaker_start_s", "stt", "llm_model"])
    start_piper()
    filler_wavs = [LOG_DIR / f"filler_{i}.wav" for i in range(len(FILLERS))]
    for text, path in zip(FILLERS, filler_wavs):   # made once now, so playing a filler later costs no time
        make_wav(text, path)

    print(f"Ready (LLM: {LLM_MODEL}). Say or type 'stop chat' to end. Log: {log_path}\n")
    gaze = start_gaze(args.gaze_device) if args.gaze else None   # camera on, only with --gaze
    if args.gaze:
        threading.Thread(target=read_doa, daemon=True).start()   # sound direction, only with --gaze
    start_web()                                                  # live status page

    history, turn, waiting = [], 0, args.gaze
    asked = False                          # did we already ask STILL_THERE in this silence?
    told_english = False                   # said "only English" to this person already?

    try:
        while True:
            if waiting:
                status["state"] = "WAITING"
                wait_for_look()
                greeting = random.choice(GREETINGS)   # pick one greeting at random each time
                print(f"robot: {greeting}\n", flush=True)
                status.update(step2=None, voice_deg=None, face_deg=None, lips=None, turns=[], times={})   # new partner: clear the page
                told_english = False                       # new person: may need the English hint once

                status["state"] = "SPEAKING"
                robot_says(greeting, wav_path)
                last_talk = time.perf_counter()   # robot just spoke
                waiting = False
            look = lips = None                             # camera facts for this sentence (typed text: none)

            if controls["typed"]:                          # a sentence typed on the web page
                user_text, controls["typed"] = controls["typed"], None
                speech_end, whisper_s = time.perf_counter(), 0.0

            elif args.text_input:
                user_text = input("you: ").strip()
                speech_end, whisper_s = time.perf_counter(), 0.0
            else:
                status["state"] = "LISTENING" # green light: speak now
                mic = start_mic()
                audio, speech_end = record_sentence(mic, vad, give_up=lambda: controls["typed"] or (partner_gone() if args.gaze else None))   # stop listening if typed or partner gone
                audio_end = time.perf_counter()   # when the recording really ended (lip check needs exact times)

                stop_mic(mic)                     # do not listen while we think and speak
                status["state"] = "THINKING"
                if audio is None:
                    print("Microphone stopped (is the ReSpeaker connected?)")
                    break
                if len(audio) == 0:            # partner_gone() stopped the listening
                    if controls["typed"]:
                        continue                  # typed on the page: handled at the top of the loop

                    reason = partner_gone()
                    if reason == "not engaged" and not asked:
                        print(f"robot: {STILL_THERE}\n", flush=True)
                        status["state"] = "SPEAKING"
                        robot_says(STILL_THERE, wav_path)
                        last_talk, asked = time.perf_counter(), True   # give them 30 s more
                        continue
                    print(f"robot: {GOODBYE}   [partner unlocked: {reason}]\n", flush=True)
                    status["state"] = "SPEAKING"
                    robot_says(GOODBYE, wav_path)
                    if gaze:
                        gaze.stdin.write("UNLOCK\n"); gaze.stdin.flush()   # people_gaze frees the partner's ID
                    history, waiting, asked = [], True, False              # back to waiting for a look (always)
                    continue
                if args.gaze:                     # lips FIRST, Whisper only for the partner (sequential, 16 Nov)
                    ask_lips(gaze, audio, audio_end - len(audio) / SAMPLE_RATE, speech_end + 0.1)
                    speech_start = speech_end - len(audio) / SAMPLE_RATE    # roughly when this sentence started
                    ok, v, f = voice_from_partner(speech_start, speech_end)
                    look = look_pct(speech_start, speech_end)
                    print(f"     Looking at robot: {look if look is not None else 'no data'} % of the sentence", flush=True)

                    status.update(step2=ok, voice_deg=v, face_deg=f)      # for the web page
                    t_lip = time.perf_counter()
                    lips = lip_result()                   # wait for the lip check (Whisper not started yet)
                    status["lips"] = lips
                    print(f"     Lips: {lips if lips is not None else 'no data'} % speaking  "
                          f"(sentence {len(audio) / SAMPLE_RATE:.1f} s, lip wait {time.perf_counter() - t_lip:.2f} s)", flush=True)
                    # rule B: the lips decide; only without lip data does the sound direction (Step 2) decide
                    partner = (lips >= LIP_MIN_PCT) if lips is not None else (ok is not False)
                    if not partner:
                        print("     -> ignored (not the partner), no Whisper\n", flush=True)
                        status["turns"] = (status["turns"] + [["ignored", "(not transcribed)"]])[-10:]   # page shows it grey
                        continue                          # no answer, and it does not count as the partner talking

                t = time.perf_counter()
                if lang_model:                        # option A: tiny only detects the language, base.en writes the text
                    lang, lang_p, all_p = lang_model.detect_language(audio)
                segments, info = model.transcribe(audio, language="en" if lang_model else None, beam_size=5,
                                               condition_on_previous_text=False, vad_filter=False,
                                               temperature=0.0,
                                               hotwords="Nila FAU Erlangen FAPS robot")   # hard words: names + places (fixes "Nilla", "Lanjuan")
                if not lang_model:                    # multilingual model detects the language itself
                    lang, lang_p, all_p = info.language, info.language_probability, info.all_language_probs
                # English's own probability (accented English: top guess may be wrong).
                # English-only models (base.en, small.en) give no list (all_p = None): then use the detected language itself
                en_p = dict(all_p).get("en", 0.0) if all_p else (lang_p if lang == "en" else 0.0)
                print(f"     Language: {lang} {lang_p:.2f} | en {en_p:.2f}", flush=True)
                if en_p < 0.15:                       # not English (0.15 from our tests)
                    print(f"     -> not English ({lang}, en {en_p:.2f}), ignored\n", flush=True)
                    if not told_english and lang_p >= 0.5:      # clearly another language: say once what to do
                        status["state"] = "SPEAKING"
                        robot_says(ENGLISH_ONLY, wav_path)
                        told_english = True
                    continue
                if lang != "en" and not lang_model:  # accented English: top guess was another language -> write it in English
                    print(f"     (top guess {lang}, en {en_p:.2f}: transcribed again as English)", flush=True)
                    segments, info = model.transcribe(audio, language="en", beam_size=5,
                                                      condition_on_previous_text=False, vad_filter=False,
                                                      temperature=0.0,
                                                      hotwords="Nila FAU Erlangen FAPS robot")   # same hard words


                segments = [s for s in segments if s.compression_ratio < 2.4]   # drop repeated hallucinations ("All right. All right. ...")
                
                segments = [s for s in segments if s.avg_logprob > -1.0]        # drop low-confidence text (other language, mumbling)
                
                segments = [s for s in segments if s.no_speech_prob < 0.5]      # drop noise: Whisper itself says "no speech" (speech <= 0.15, tap/cough 0.54-0.72 in our test)
                # noise can make Whisper "say" the hotwords themselves -> drop text made only of hotwords
                HOT = {"nila", "fau", "erlangen", "faps", "robot"}
                if segments and set(re.sub(r"[^a-z ]", " ", " ".join(s.text for s in segments).lower()).split()) <= HOT:
                    segments = []
                user_text = " ".join(s.text.strip() for s in segments).strip()
                whisper_s = time.perf_counter() - t
          
                status["times"] = {"whisper": round(whisper_s, 2)}   # show Whisper time for every sentence, also ignored ones

                print(f"you: {user_text or '(nothing)'}")

            if not user_text:
            
                continue
            asked, last_talk = False, time.perf_counter()   # the person spoke -> engaged again

            if is_stop(user_text):
                print("Stop phrase heard. Bye.")
                if not args.gaze:
                    break
                history, waiting = [], True       # new person, new conversation
                continue
                

            turn += 1
            t = time.perf_counter()
            filler = None if controls["mute"] else subprocess.Popen(
                ["aplay", "-q", "-D", echo_demo.PLAYBACK_DEVICE, str(random.choice(filler_wavs))])   # "Hmm." plays while the LLM works (no waiting)

            hint = hint_line(look, lips)
            print(f"     to LLM: {user_text}{hint}", flush=True)
            answer = ask_llm(history + [{"role": "user", "content": user_text + hint}])
            llm_s = time.perf_counter() - t
            if "[IGNORE]" in answer:              # partner talks to someone else -> robot stays quiet
                if filler:
                    filler.wait()
                print("     -> [IGNORE] not for the robot, stays quiet\n", flush=True)
                status["turns"] = (status["turns"] + [["ignored", user_text]])[-10:]
                continue

            if "[REPEAT]" in answer:
                answer = REPEAT_TEXT              # unclear input: ask again, keep it out of the history
            else:
                history += [{"role": "user", "content": user_text},
                            {"role": "assistant", "content": answer}]
            end_chat = "[END]" in answer          # person said goodbye
            answer = answer.replace("[END]", "").strip()

            print(f"robot: {answer}")
            status["turns"] = (status["turns"] + [["you", user_text], ["robot", answer]])[-10:]   # last 5 turns

            status["state"] = "SPEAKING"
            if filler:
                filler.wait()                     # let "Hmm." finish before the answer starts

            synth_s, playback_start = robot_says(answer, wav_path)
            last_talk = time.perf_counter()   # robot finished speaking
            told_english = False              # understood again: next misunderstanding may get the hint again

            end_to_speaker = playback_start - speech_end
            print(f"     whisper {whisper_s:.2f} s | llm {llm_s:.2f} s | piper {synth_s:.2f} s | "
                  f"speech end -> robot speaks {end_to_speaker:.2f} s\n", flush=True)
            status["times"] = {"whisper": round(whisper_s, 2), "llm": round(llm_s, 2),
                        "piper": round(synth_s, 2), "total": round(end_to_speaker, 2)}

            log.writerow([turn, user_text, answer, f"{whisper_s:.3f}", f"{llm_s:.3f}", f"{synth_s:.3f}",
                          f"{end_to_speaker:.3f}", source, LLM_MODEL])
            log_file.flush()
            if end_chat:                          # goodbye said -> unlock, wait for the next look
                print("     [partner unlocked: goodbye]\n", flush=True)
                if gaze:
                    gaze.stdin.write("UNLOCK\n"); gaze.stdin.flush()       # people_gaze frees the partner's ID
                if not args.gaze:
                    break
                history, waiting, asked = [], True, False

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        log_file.close()
        if gaze:
            gaze.terminate()   # camera off when the demo ends


if __name__ == "__main__":
    main()
