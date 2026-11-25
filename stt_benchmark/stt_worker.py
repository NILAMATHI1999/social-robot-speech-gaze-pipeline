"""One speech-to-text model as a worker process (runs in that model's own venv).

Reads WAV paths (16 kHz mono) from stdin, one per line, and prints one JSON line per file:
{"text": ..., "seconds": ...}. Used by live_compare.py.

  python stt_worker.py --engine whisper  --model large-v3-turbo   (~/ament_ws/.venv)
  python stt_worker.py --engine parakeet                          (~/stt_parakeet_venv)
"""

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
PARAKEET_DIR = Path.home() / "stt_models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"


def load_wav(path):
    with wave.open(path) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0


def make_transcriber(engine, model_name, device, compute_type):
    """Load the model once; return a function audio -> text."""
    if engine == "whisper":
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device=device, compute_type=compute_type)

        def run(audio):
            segments, _ = model.transcribe(audio, language="en", beam_size=5,
                                           condition_on_previous_text=False, vad_filter=False,
                                           temperature=0.0,)  # expected words      )
            return " ".join(s.text.strip() for s in segments).strip()
        return run

    if engine == "parakeet":
        import sherpa_onnx
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(PARAKEET_DIR / "encoder.int8.onnx"),
            decoder=str(PARAKEET_DIR / "decoder.int8.onnx"),
            joiner=str(PARAKEET_DIR / "joiner.int8.onnx"),
            tokens=str(PARAKEET_DIR / "tokens.txt"),
            num_threads=4, sample_rate=SAMPLE_RATE, feature_dim=80,
            decoding_method="greedy_search", model_type="nemo_transducer")

        def run(audio):
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            recognizer.decode_stream(stream)
            return stream.result.text.strip()
        return run

    raise ValueError(f"unknown engine {engine}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["whisper", "parakeet"])
    parser.add_argument("--model", default="base.en", help="Whisper model name (whisper engine only)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()

    run = make_transcriber(args.engine, args.model, args.device, args.compute_type)
    run(np.zeros(SAMPLE_RATE, dtype=np.float32))          # warm-up, so the first sentence is not slow
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        started = time.perf_counter()
        try:
            text = run(load_wav(path))
            error = ""
        except Exception as exc:                            # report, keep the worker alive
            text, error = "", repr(exc)
        print(json.dumps({"text": text, "seconds": round(time.perf_counter() - started, 3),
                          "error": error}), flush=True)


if __name__ == "__main__":
    main()

