import sys
import time
import types
import wave
from pathlib import Path

import numpy as np
import queue, subprocess, threading
from faster_whisper.vad import get_vad_model   # Silero VAD, same as the MVP (echo_demo.py)


# BLOCK 1: load the streaming code.
# whisper_online.py imports librosa and soundfile at the top, but we never use the parts that need them.
# These two empty fake modules let the import work, so we install nothing and numpy stays at 1.26.4.
sys.modules["librosa"] = types.ModuleType("librosa")
sys.modules["soundfile"] = types.ModuleType("soundfile")
# Tell Python where the cloned repo is, then take the two classes we need.
sys.path.insert(0, str(Path.home() / "projects" / "whisper_streaming"))
from whisper_online import FasterWhisperASR, OnlineASRProcessor

# BLOCK 2: settings.2
CHUNK_S = 2.0   # give the model new audio every 2 s
RATE = 16000    # samples per second of our clips
END_PAUSE_S = 0.8     # professor: 0.8 s of silence = end of the sentence (test 1.2 later)
VAD_THRESHOLD = 0.5   # same as the MVP


# BLOCK 3: run Whisper on the laptop CPU.
# The repo's load_model always uses the GPU (cuda, float16), which fails on the laptop.
# This small class keeps everything else and replaces only that one function.
class Asr(FasterWhisperASR):
    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel
        return WhisperModel(modelsize, device="cpu", compute_type="int8")
    # Same call as the repo, with 2 changes:
    # condition_on_previous_text=False: Whisper doesn't read its own earlier text, which stopped the repeat loops.
    # hallucination_silence_threshold=2.0: skips invented text around silence longer than 2 s.
    def transcribe(self, audio, init_prompt=""):
        segments, _ = self.model.transcribe(audio, language=self.original_language, initial_prompt=init_prompt,
                                            beam_size=5, word_timestamps=True, condition_on_previous_text=False,
                                            hallucination_silence_threshold=2.0, **self.transcribe_kargs)
        return list(segments)


# BLOCK 4: microphone thread.
# arecord streams the ReSpeaker (6 channels). This thread reads it the whole time and puts
# each 0.1 s piece into a queue, so no audio is lost while Whisper is busy with a pass.
# WHO DOES "waiting": this thread. It fills the queue all the time, also while Whisper is busy.

MIC = "hw:CARD=ArrayUAC10,DEV=0"   # same device as the MVP (echo_demo.py)
PIECE = 1600                        # 0.1 s of audio
mic_queue = queue.Queue()
vad = get_vad_model()
first_voice = None   # clock time when the person started speaking (for the first-word time)
last_voice = None    # clock time of the last piece with voice (for the 0.8 s pause)

def mic_thread():
    mic = subprocess.Popen(["arecord", "-q", "-D", MIC, "-f", "S16_LE", "-r", str(RATE), "-c", "6", "-t", "raw"],
                           stdout=subprocess.PIPE)
    n_bytes = PIECE * 6 * 2          # 6 channels, 2 bytes per sample
    while True:
        raw = mic.stdout.read(n_bytes)
        if len(raw) < n_bytes:       # mic stopped
            break
        audio = np.frombuffer(raw, dtype="<i2").reshape(-1, 6)[:, 0]   # channel 0, like the robot
        global first_voice, last_voice
        piece = audio.astype(np.float32) / 32768.0
        # VAD on this 0.1 s piece (Silero needs multiples of 512 samples, so check the first 1536)
        if float(np.max(vad(piece[:1536]))) >= VAD_THRESHOLD:
            last_voice = time.perf_counter()
            if first_voice is None:
                first_voice = last_voice      # speech just started
        mic_queue.put(piece)


# BLOCK 5: prepare.
# sys.argv[1] is the model name from the command line (base.en for live on the laptop).
asr = Asr("en", sys.argv[1])
asr.transcribe_kargs["temperature"] = 0.0   # our setting: no slow retries
asr.use_vad()   # our setting: use VAD to skip silence
online = OnlineASRProcessor(asr, buffer_trimming=("segment", 8))   # trim the buffer once it passes 8 s
threading.Thread(target=mic_thread, daemon=True).start()   # start reading the mic in the background
recorded = []   # all live audio, saved at the end so we can re-run it with other settings
print(f"speak now, new audio every {CHUNK_S} s, Ctrl+C to stop", flush=True)

# BLOCK 6: live loop.
# Wait until one step of audio is in the queue, then take ALL the waiting audio.
# "waiting" = audio that piled up since the last pass. If it stays near the step, Whisper keeps up.
# If it keeps growing, Whisper is falling behind the speaker (this is the real delay).
shown = False   # first word of this sentence already printed?
start = time.perf_counter()
try:
    while True:
        while mic_queue.qsize() < CHUNK_S * 10:   # 10 pieces of 0.1 s = 1 s
            if last_voice and time.perf_counter() - last_voice >= END_PAUSE_S:
                break                             # 0.8 s silence: don't wait for a full step
            time.sleep(0.05)

        pieces = [mic_queue.get() for _ in range(mic_queue.qsize())] or [np.zeros(0, np.float32)]
        waiting = len(pieces) / 10         # waiting = how much audio the mic thread put in the queue since the last pass (10 pieces = 1 s)
        chunk = np.concatenate(pieces)
        recorded.append(chunk)   # keep it for the WAV file
        online.insert_audio_chunk(chunk)
        t = time.perf_counter()          # WHO DOES "took": Whisper. process_iter() is one pass; the clock before and after it gives "took".
        _, _, text = online.process_iter()
        print(f"{time.perf_counter() - start:5.1f} s | waiting {waiting:4.1f} s | took {time.perf_counter() - t:.2f} s | {text}", flush=True)
        # FIRST WORD: time from speech start (VAD) to the first confirmed words on the screen.
        # Example j_commands_1: started speaking at 0.8 s, "Hello" printed at 6.9 s -> 6.1 s (6.9 - 0.8) to first word.
        # It is NOT a word you said at that time. It is how long you waited until the first words appeared on the screen after you speak.
        if text and first_voice and not shown:
            print(f"        first word {time.perf_counter() - first_voice:.1f} s after speech start", flush=True)
            shown = True
        # END OF SPEECH: 0.8 s silence. The pass above was the last one; finish() prints the held-back words
        # (before, only Ctrl+C did this). Then reset for the next sentence.
        # Example: "I don't know what" came while speaking; END OF SPEECH | "is happening." = the held-back end, after you stopped.
        if last_voice and time.perf_counter() - last_voice >= END_PAUSE_S:
            _, _, rest = online.finish()
            print(f"        END OF SPEECH ({time.perf_counter() - last_voice:.1f} s after the last voice) | {rest}", flush=True)
            online.init()
            first_voice, last_voice, shown = None, None, False

# BLOCK 7: Ctrl+C pressed.
# finish() returns the last words that were still waiting for confirmation.
except KeyboardInterrupt:
    _, _, text = online.finish()
    print(f"  end   | {text}")
    # Save everything we heard as a 1-channel WAV, e.g. results/live_c_long_1.wav
    out = Path("results") / f"live_{sys.argv[2]}.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((np.concatenate(recorded) * 32768).clip(-32768, 32767).astype("<i2").tobytes())
    print(f"saved {out}")
