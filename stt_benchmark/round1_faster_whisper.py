"""Round 1: compare faster-whisper models on the recorded STT test set.

Every model runs on the same clips with the same settings, once without VAD
and once with Silero VAD in front (like the robot pipeline). Each model runs
in its own process so its peak RAM can be measured cleanly.

Outputs (in results/):
  round1_clips_<time>.csv    one row per model, VAD mode and clip
  round1_summary_<time>.csv  one row per model and VAD mode (sheet columns)

Run:
  source ~/ament_ws/.venv/bin/activate
  python round1_faster_whisper.py                    # everything (30-60 min on CPU)
  python round1_faster_whisper.py --models base.en --limit 3   # quick check
  python round1_faster_whisper.py --device cuda --compute-type float16   # Jetson GPU
"""

import argparse
import csv
import multiprocessing as mp
import resource
import statistics
import time
import wave
from datetime import datetime
from pathlib import Path

import jiwer
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from whisper_normalizer.english import EnglishTextNormalizer

TESTSET = Path(__file__).resolve().parent.parent / "stt_testset"
RESULTS = Path(__file__).resolve().parent / "results"
CHANNEL = 0                 # same channel the robot pipeline uses
SAMPLE_RATE = 16000

MODELS = ["base.en", "small.en", "medium.en", "distil-whisper/distil-large-v3.5-ct2", "large-v3-turbo"]
DEVICE = "cpu"
COMPUTE_TYPE = "int8"       # Jetson later: "int8_float16"

# Same decoding settings for every model.
DECODE = dict(language="en", beam_size=5, condition_on_previous_text=False, vad_filter=False)

# Same Silero settings as silero_streaming_node.py.
SILERO = VadOptions(threshold=0.5, min_speech_duration_ms=250,
                    min_silence_duration_ms=500, speech_pad_ms=200)

SPEECH_SETS = ["clean", "background", "background_loud", "cafe_speech_loud"]
EMPTY_SETS = ["silence", "noise", "noise_loud"]

normalize = EnglishTextNormalizer()


# ----------------------------------------------------------------------------
# Test set
# ----------------------------------------------------------------------------
def load_clip(wav_path):
    """Return channel 0 of a 6-channel WAV as float32 in [-1, 1]."""
    with wave.open(str(wav_path)) as w:
        raw = w.readframes(w.getnframes())
        channels = w.getnchannels()
    audio = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)[:, CHANNEL]
    return audio.astype(np.float32) / 32768.0


def load_testset(limit=None):
    """List of (clip name, set name, audio, reference text)."""
    clips = []
    for wav_path in sorted(TESTSET.glob("*.wav")):
        name = wav_path.stem                       # e.g. background_loud_07
        set_name = name.rsplit("_", 1)[0]           # e.g. background_loud
        reference = wav_path.with_suffix(".txt").read_text().strip()
        clips.append((name, set_name, load_clip(wav_path), reference))
    if limit:
        # Keep the first `limit` clips of every set.
        kept, count = [], {}
        for clip in clips:
            count[clip[1]] = count.get(clip[1], 0) + 1
            if count[clip[1]] <= limit:
                kept.append(clip)
        clips = kept
    return clips


# ----------------------------------------------------------------------------
# VAD and transcription
# ----------------------------------------------------------------------------
def silero_speech(audio):
    """Speech parts found by Silero, joined together (empty array if none)."""
    stamps = get_speech_timestamps(audio, SILERO)
    if not stamps:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([audio[s["start"]:s["end"]] for s in stamps])


def run_model(model_name, vad_modes, clips, queue, device, compute_type):
    """Child process: load one model, transcribe all clips, report rows + peak RAM."""
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    rows = []
    for vad in vad_modes:
        for name, set_name, audio, reference in clips:
            started = time.perf_counter()
            if vad == "silero":
                audio = silero_speech(audio)
            if len(audio) == 0:
                text = ""                              # VAD removed everything
            else:
                segments, _ = model.transcribe(audio, **DECODE)
                text = " ".join(s.text.strip() for s in segments).strip()
            seconds = time.perf_counter() - started
            rows.append(dict(model=model_name, device=device, compute_type=compute_type,
                             vad=vad, clip=name, set=set_name,
                             reference=reference, hypothesis=text, seconds=round(seconds, 3)))
            print(f"  {model_name:40} {vad:6} {name:22} {seconds:5.2f} s  {text[:60]}", flush=True)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # kB -> MB on Linux
    queue.put((rows, peak_mb))


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def set_wer(rows):
    """Corpus WER (%) over all clips of a set, after Whisper normalisation."""
    refs = [normalize(r["reference"]) for r in rows]
    hyps = [normalize(r["hypothesis"]) for r in rows]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return None
    return 100 * jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs])


def hallucinations(rows):
    """Clips where the model wrote any word although nobody spoke to the robot."""
    return sum(1 for r in rows if normalize(r["hypothesis"]).strip())


def summarize(rows, peak_mb):
    out = dict(model=rows[0]["model"], device=rows[0]["device"],
               compute_type=rows[0]["compute_type"], vad=rows[0]["vad"])
    for s in SPEECH_SETS:
        wer = set_wer([r for r in rows if r["set"] == s])
        out[f"wer_{s}"] = "" if wer is None else round(wer, 1)
    for s in EMPTY_SETS:
        set_rows = [r for r in rows if r["set"] == s]
        out[f"halluc_{s}"] = f"{hallucinations(set_rows)}/{len(set_rows)}" if set_rows else ""
    clean = [r["seconds"] for r in rows if r["set"] == "clean"]
    out["latency_median_s"] = round(statistics.median(clean), 2) if clean else ""
    out["peak_ram_mb"] = round(peak_mb)
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--vad", choices=["none", "silero", "both"], default="both")
    parser.add_argument("--limit", type=int, help="only the first N clips of every set (quick test)")
    parser.add_argument("--device", default=DEVICE, help="cpu (laptop) or cuda (Jetson GPU)")
    parser.add_argument("--compute-type", default=COMPUTE_TYPE,
                        help="int8 on CPU; float16 or int8_float16 on the Jetson GPU")
    args = parser.parse_args()

    vad_modes = ["none", "silero"] if args.vad == "both" else [args.vad]
    clips = load_testset(args.limit)
    print(f"{len(clips)} clips, models: {args.models}, VAD: {vad_modes}, "
          f"device: {args.device}, compute type: {args.compute_type}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.device}_{args.compute_type}_{stamp}"
    clip_csv = RESULTS / f"round1_clips_{tag}.csv"
    summary_csv = RESULTS / f"round1_summary_{tag}.csv"

    all_rows, summaries = [], []
    ctx = mp.get_context("spawn")
    for model_name in args.models:
        print(f"\n=== {model_name} ===", flush=True)
        queue = ctx.Queue()
        proc = ctx.Process(target=run_model, args=(model_name, vad_modes, clips, queue,
                                                       args.device, args.compute_type))
        proc.start()
        rows, peak_mb = queue.get()
        proc.join()
        all_rows += rows
        for vad in vad_modes:
            summaries.append(summarize([r for r in rows if r["vad"] == vad], peak_mb))

        # Save after every model, so a crash later keeps the finished results.
        with open(clip_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0]))
            w.writeheader()
            w.writerows(summaries)

    print(f"\nPer clip: {clip_csv}\nSummary:  {summary_csv}\n")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
