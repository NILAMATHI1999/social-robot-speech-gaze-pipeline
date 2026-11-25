"""Piper TTS worker: loads the voice once, then turns each request into a WAV file.

Started by echo_demo.py / chat_demo.py with the Python that has piper-tts (own venv on the Jetson).
Loading the voice takes about 2 s; with the voice kept in memory one sentence takes about 0.3 s.
Protocol: one JSON line per request on stdin {"text": ..., "wav": ...}, answer "done" on stdout.
"""

import json
import sys
import wave

from piper import PiperVoice

voice = PiperVoice.load(sys.argv[1])
print("ready", flush=True)

for line in sys.stdin:
    job = json.loads(line)
    with wave.open(job["wav"], "wb") as wav_file:
        voice.synthesize_wav(job["text"], wav_file)
    print("done", flush=True)

