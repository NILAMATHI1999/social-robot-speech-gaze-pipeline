"""Round 2: Parakeet TDT 0.6B v2 (sherpa-onnx, int8) on the same recorded STT test set.

Same clips, same Silero cuts (results/silero_segments.json from export_silero_segments.py)
and the same scoring as round1_faster_whisper.py, so the numbers compare directly.

Run (separate venv, no torch):
  source ~/stt_parakeet_venv/bin/activate
  python round2_parakeet.py                 # everything
  python round2_parakeet.py --limit 2       # quick check
"""

import argparse
import csv
import json
import resource
import statistics
import time
import wave
from datetime import datetime
from pathlib import Path

import jiwer
import numpy as np
import sherpa_onnx
from whisper_normalizer.english import EnglishTextNormalizer

TESTSET = Path.home() / "projects/social-robot-speech-gaze-pipeline/stt_testset"
RESULTS = Path(__file__).resolve().parent / "results"
MODEL_DIR = Path.home() / "stt_models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
MODEL_NAME = "parakeet-tdt-0.6b-v2-int8"
CHANNEL = 0                 # same channel the robot pipeline uses
SAMPLE_RATE = 16000
NUM_THREADS = 4

SPEECH_SETS = ["clean", "background", "background_loud", "cafe_speech_loud"]
EMPTY_SETS = ["silence", "noise", "noise_loud"]

normalize = EnglishTextNormalizer()


def load_clip(wav_path):
    """Return channel 0 of a 6-channel WAV as float32 in [-1, 1]."""
    with wave.open(str(wav_path)) as w:
        raw = w.readframes(w.getnframes())
        channels = w.getnchannels()
    audio = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)[:, CHANNEL]
    return audio.astype(np.float32) / 32768.0


def load_testset(limit=None):
    clips, count = [], {}
    for wav_path in sorted(TESTSET.glob("*.wav")):
        name = wav_path.stem
        set_name = name.rsplit("_", 1)[0]
        count[set_name] = count.get(set_name, 0) + 1
        if limit and count[set_name] > limit:
            continue
        reference = wav_path.with_suffix(".txt").read_text().strip()
        clips.append((name, set_name, load_clip(wav_path), reference))
    return clips


def silero_speech(audio, stamps):
    """Speech parts from the saved Silero segments, joined (empty array if none)."""
    if not stamps:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([audio[s["start"]:s["end"]] for s in stamps])


def set_wer(rows):
    pairs = [(normalize(r["reference"]), normalize(r["hypothesis"])) for r in rows]
    pairs = [p for p in pairs if p[0].strip()]
    if not pairs:
        return None, None
    out = jiwer.process_words([p[0] for p in pairs], [p[1] for p in pairs])
    n = out.hits + out.substitutions + out.deletions
    return 100 * out.wer, 100 * out.insertions / n


def summarize(rows, peak_mb):
    out = dict(model=MODEL_NAME, device="cpu", compute_type="int8", vad=rows[0]["vad"])
    for s in SPEECH_SETS:
        wer, extra = set_wer([r for r in rows if r["set"] == s])
        out[f"wer_{s}"] = "" if wer is None else round(wer, 1)
        out[f"extra_{s}"] = "" if extra is None else round(extra, 1)
    for s in EMPTY_SETS:
        set_rows = [r for r in rows if r["set"] == s]
        halluc = sum(1 for r in set_rows if normalize(r["hypothesis"]).strip())
        out[f"halluc_{s}"] = f"{halluc}/{len(set_rows)}" if set_rows else ""
    clean = [r["seconds"] for r in rows if r["set"] == "clean"]
    out["latency_median_s"] = round(statistics.median(clean), 2) if clean else ""
    out["peak_ram_mb"] = round(peak_mb)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only the first N clips of every set (quick test)")
    args = parser.parse_args()

    segments = json.loads((RESULTS / "silero_segments.json").read_text())
    clips = load_testset(args.limit)
    print(f"{len(clips)} clips, model: {MODEL_NAME}")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(MODEL_DIR / "encoder.int8.onnx"),
        decoder=str(MODEL_DIR / "decoder.int8.onnx"),
        joiner=str(MODEL_DIR / "joiner.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        num_threads=NUM_THREADS,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
    )

    rows = []
    for vad in ["none", "silero"]:
        for name, set_name, audio, reference in clips:
            started = time.perf_counter()
            if vad == "silero":
                audio = silero_speech(audio, segments[name])
            if len(audio) == 0:
                text = ""                              # VAD removed everything
            else:
                stream = recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, audio)
                recognizer.decode_stream(stream)
                text = stream.result.text.strip()
            seconds = time.perf_counter() - started
            rows.append(dict(model=MODEL_NAME, device="cpu", compute_type="int8", vad=vad,
                             clip=name, set=set_name, reference=reference, hypothesis=text,
                             seconds=round(seconds, 3)))
            print(f"  {vad:6} {name:22} {seconds:5.2f} s  {text[:60]}", flush=True)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    summaries = [summarize([r for r in rows if r["vad"] == v], peak_mb) for v in ["none", "silero"]]

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clip_csv = RESULTS / f"round2_parakeet_clips_cpu_int8_{stamp}.csv"
    summary_csv = RESULTS / f"round2_parakeet_summary_cpu_int8_{stamp}.csv"
    with open(clip_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)

    print(f"\nPer clip: {clip_csv}\nSummary:  {summary_csv}\n")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
