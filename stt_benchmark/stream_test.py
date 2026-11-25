import sys
import time
import types
import wave
from pathlib import Path

import numpy as np

# BLOCK 1: load the streaming code.
# whisper_online.py imports librosa and soundfile at the top, but we never use the parts that need them.
# These two empty fake modules let the import work, so we install nothing and numpy stays at 1.26.4.
import faster_whisper   # load before the fake modules (on the Jetson, transformers checks for librosa)

sys.modules["librosa"] = types.ModuleType("librosa")
sys.modules["soundfile"] = types.ModuleType("soundfile")
# Tell Python where the cloned repo is, then take the two classes we need.
sys.path.insert(0, str(Path.home() / "projects" / "whisper_streaming"))
from whisper_online import FasterWhisperASR, OnlineASRProcessor

# BLOCK 2: settings.2
CHUNK_S = 2.0   # give the model new audio every 2 s
RATE = 16000    # samples per second of our clips


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


# BLOCK 4: read a WAV clip.
# Our clips have 6 ReSpeaker channels; [:, 0] keeps channel 0, the same one the robot uses.
# Whisper wants numbers between -1 and 1, so divide by 32768.
def load_clip(path):
    with wave.open(str(path)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, w.getnchannels())[:, 0]
    return audio.astype(np.float32) / 32768.0


# BLOCK 5: prepare.
# sys.argv[1] is the clip path from the command line.
# OnlineASRProcessor is the sliding-window part: it keeps the audio buffer and decides which text is final.
audio = load_clip(sys.argv[1])
asr = Asr("en", sys.argv[2])   # model name from the command line
asr.transcribe_kargs["temperature"] = 0.0   # our setting: no slow retries
asr.use_vad()   # our setting: use VAD to skip silence
online = OnlineASRProcessor(asr, buffer_trimming=("segment", 8))   # trim the buffer once it passes 8 s

print(f"clip {len(audio) / RATE:.1f} s, new audio every {CHUNK_S} s")
step = int(CHUNK_S * RATE)   # 1 s = 16000 samples

# BLOCK 6: pretend the clip is live, 1 s at a time.
# insert_audio_chunk adds the new second to the buffer.
# process_iter re-reads the buffer with Whisper and returns only newly confirmed text
# (words that came out the same in two passes in a row), so it is often empty.
# "took" is the time for one pass. If it is longer than CHUNK_S, it can't keep up live.
for start in range(0, len(audio), step):
    online.insert_audio_chunk(audio[start:start + step])
    t = time.perf_counter()
    _, _, text = online.process_iter()
    print(f"{(start + step) / RATE:5.1f} s | took {time.perf_counter() - t:.2f} s | {text}")

# BLOCK 7: audio finished.
# finish() returns the last words that were still waiting for confirmation.
t = time.perf_counter()
_, _, text = online.finish()
print(f"  end   | took {time.perf_counter() - t:.2f} s | {text}")
