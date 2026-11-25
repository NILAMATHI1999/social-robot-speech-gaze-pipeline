"""MVP step 2: echo demo. ReSpeaker -> Silero VAD (0.8 s pause) -> faster-whisper -> Piper repeats.

No LLM. Shows the speech-to-text delay and how well the pause detection works.
The microphone is stopped while Piper speaks, so the robot does not hear itself.

Run (laptop, venv with faster-whisper + piper):
  "/home/nila/thesis/old repos/subash-social-robot/.venv-stt/bin/python" echo_demo.py --model base.en
Jetson later:
  python echo_demo.py --model large-v3-turbo --device cuda --compute-type float16
Stop: Ctrl+C, or say "stop echo".
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model
from difflib import SequenceMatcher

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
ALSA_DEVICE = "hw:CARD=ArrayUAC10,DEV=0"
CHANNELS = 6
CHANNEL = 0                       # same channel the robot pipeline uses
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1536              # 96 ms, a multiple of Silero's 512-sample window
CHUNK_S = CHUNK_SAMPLES / SAMPLE_RATE

VAD_THRESHOLD = 0.5               # same as silero_streaming_node.py
END_PAUSE_S = float(os.environ.get("END_PAUSE_S", 0.8))   # professor: 0.8 s pause = end of the sentence
USE_SMART_TURN = os.environ.get("USE_SMART_TURN") == "1"   # Smart Turn decides "finished?" instead of a fixed 0.8 s pause
SMART_TURN_CHECK_S = 0.3          # after 0.3 s of silence, ask Smart Turn (Pipecat: run it when the VAD hears silence)
MAX_PAUSE_S = 2.0                 # a thinking pause can be at most 2 s, then the sentence ends anyway
if USE_SMART_TURN:
    from smart_turn import turn_finished
PRE_ROLL_S = 0.3                  # keep audio from before the speech start (first word)
MIN_SPEECH_S = 0.25               # shorter blips are ignored
MAX_SPEECH_S = 20.0

PIPER_VOICE = os.environ.get("PIPER_VOICE", str(Path(__file__).resolve().parent / "piper_voices" / "en_US-lessac-medium.onnx"))
PIPER_PYTHON = os.environ.get("PIPER_PYTHON", sys.executable)     # Jetson: Piper has its own venv
PLAYBACK_DEVICE = os.environ.get("PLAYBACK_DEVICE", "default")    # Jetson: ReSpeaker 3.5 mm jack
STOP_PHRASES = ("stop echo",)
LOG_DIR = Path(__file__).resolve().parent / "logs"


# ----------------------------------------------------------------------------
# Microphone
# ----------------------------------------------------------------------------
def start_mic():
    return subprocess.Popen(
        ["arecord", "-q", "-D", ALSA_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
         "-c", str(CHANNELS), "-t", "raw"],
        stdout=subprocess.PIPE,
    )


def read_chunk(mic):
    """Next 96 ms of channel 0 as float32 in [-1, 1], or None if the stream ended."""
    n_bytes = CHUNK_SAMPLES * CHANNELS * 2
    raw = mic.stdout.read(n_bytes)
    if len(raw) < n_bytes:
        return None
    audio = np.frombuffer(raw, dtype="<i2").reshape(-1, CHANNELS)[:, CHANNEL]
    return audio.astype(np.float32) / 32768.0


def stop_mic(mic):
    mic.terminate()
    mic.wait()


# ----------------------------------------------------------------------------
# One sentence: wait for speech, record until a 0.8 s pause
# ----------------------------------------------------------------------------
def record_sentence(mic, vad,give_up=None):
    """Return (audio, speech_end_time) or (None, None) if the mic stopped.
    give_up: optional function, asked while nobody speaks. If it returns something true
    (e.g. the partner left), we stop listening and return (empty audio, None)."""
    pre_roll = []
    speech = []
    in_speech = False
    silent_s = 0.0
    checked = False
    speech_end = None

    while True:
        chunk = read_chunk(mic)
        if chunk is None:
            return None, None
        is_voice = float(np.max(vad(chunk))) >= VAD_THRESHOLD

        if not in_speech:
            pre_roll.append(chunk)
            pre_roll = pre_roll[-int(PRE_ROLL_S / CHUNK_S) - 1:]
            if is_voice:
                in_speech = True
                speech_end = time.perf_counter()      # NEW: voice just started
                speech = list(pre_roll)
                print("  ... hearing speech", flush=True)
            if give_up is not None and give_up():
                return np.zeros(0, np.float32), None
            continue

        speech.append(chunk)
        if is_voice:
            silent_s = 0.0
            speech_end = time.perf_counter()
        else:
            silent_s += CHUNK_S

        length_s = len(speech) * CHUNK_S
        if is_voice:
            checked = False                           # new speech: ask Smart Turn again at the next pause (repo advice)
        finished = False
        if USE_SMART_TURN and silent_s >= SMART_TURN_CHECK_S and not checked:
            checked = True
            finished = turn_finished(np.concatenate(speech)) > 0.5      # the whole turn so far
        end_pause = MAX_PAUSE_S if USE_SMART_TURN else END_PAUSE_S
        if finished or silent_s >= end_pause or length_s >= MAX_SPEECH_S:
            if length_s - silent_s < MIN_SPEECH_S:
                # Too short: a click or a cough. Start listening again.
                in_speech, speech, silent_s, pre_roll = False, [], 0.0, []
                continue
            return np.concatenate(speech), speech_end


# ----------------------------------------------------------------------------
# Piper
# ----------------------------------------------------------------------------

_piper = None


def start_piper():
    """Start the Piper worker once; it keeps the voice loaded (about 2 s load, then about 0.3 s per sentence)."""
    global _piper
    _piper = subprocess.Popen([PIPER_PYTHON, str(Path(__file__).resolve().parent / "piper_worker.py"), PIPER_VOICE],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, bufsize=1)
    _piper.stdout.readline()                  # wait for "ready"


def speak(text, wav_path):
    """Synthesize with Piper, then play. Returns (synthesis seconds, playback start time)."""
    if _piper is None:
        start_piper()
    started = time.perf_counter()
    _piper.stdin.write(json.dumps({"text": text, "wav": str(wav_path)}) + "\n")
    _piper.stdin.flush()
    _piper.stdout.readline()                  # "done"
    synth_s = time.perf_counter() - started
    playback_start = time.perf_counter()
    subprocess.run(["aplay", "-q", "-D", PLAYBACK_DEVICE, str(wav_path)], check=True)
    return synth_s, playback_start

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--device", default="cpu", help="cpu (laptop) or cuda (Jetson)")
    parser.add_argument("--compute-type", default="int8", help="int8 on CPU; float16 on the Jetson")
    args = parser.parse_args()

    print(f"Loading {args.model} on {args.device} ({args.compute_type}) ...", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    vad = get_vad_model()

    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"echo_{args.model.replace('/', '_')}_{args.device}_{stamp}.csv"
    wav_path = LOG_DIR / "piper_last.wav"
    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["turn", "text", "speech_s", "whisper_s", "speech_end_to_text_s",
                  "piper_synthesis_s", "speech_end_to_speaker_start_s", "model", "device"])
    start_piper()
    print(f"Ready. Speak to the robot (say 'stop echo' or press Ctrl+C to end). Log: {log_path}\n")

    turn = 0
    try:
        while True:
            mic = start_mic()
            audio, speech_end = record_sentence(mic, vad)
            stop_mic(mic)                         # do not listen while we think and speak
            if audio is None:
                print("Microphone stopped (is the ReSpeaker connected?)")
                break

            t = time.perf_counter()
            segments, _ = model.transcribe(audio, language="en", beam_size=5,
                                           condition_on_previous_text=False, vad_filter=False,
                                           temperature=0.0,)                     # no slow retries


            text = " ".join(s.text.strip() for s in segments).strip()
            whisper_s = time.perf_counter() - t
            end_to_text = time.perf_counter() - speech_end

            turn += 1
            print(f"[{turn}] heard: {text or '(nothing)'}")
            if not text:
                continue

            synth_s, playback_start = speak(text, wav_path)
            end_to_speaker = playback_start - speech_end
            print(f"     speech {len(audio) / SAMPLE_RATE:.1f} s | whisper {whisper_s:.2f} s | "
                  f"speech end -> text {end_to_text:.2f} s | piper {synth_s:.2f} s | "
                  f"speech end -> robot speaks {end_to_speaker:.2f} s\n", flush=True)
            log.writerow([turn, text, f"{len(audio) / SAMPLE_RATE:.2f}", f"{whisper_s:.3f}",
                          f"{end_to_text:.3f}", f"{synth_s:.3f}", f"{end_to_speaker:.3f}",
                          args.model, args.device])
            log_file.flush()
            
            words = text.lower().strip(" .!?")

            if any(p in words or SequenceMatcher(None, p, words).ratio() > 0.6 for p in STOP_PHRASES):
                print("Stop phrase heard. Bye.")
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        log_file.close()


if __name__ == "__main__":
    main()

