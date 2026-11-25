"""Dynamic live comparison of speech-to-text models (no Piper), two modes.

--mode switch (default): one model is active at a time; type 1/2/3 + Enter at any time to
  switch the active model while you keep talking.
--mode side: each spoken sentence goes to all models at the same time, results side by side.

The sentence is cut once with Silero VAD (0.8 s pause, like the robot), saved as WAV and sent
to the model workers (each model in its own venv, see stt_worker.py). WER per model at the end.

Run:
  source ~/ament_ws/.venv/bin/activate
  python live_compare.py --models whisper:large-v3-turbo whisper:distil-whisper/distil-large-v3.5-ct2 parakeet
  python live_compare.py --mode side --models ...
Read the sentence shown on screen; press Ctrl+C to finish.
"""

import argparse
import csv
import json
import subprocess
import sys
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mvp"))
from echo_demo import SAMPLE_RATE, record_sentence, start_mic, stop_mic   # same mic + Silero logic
from faster_whisper.vad import get_vad_model

HOME = Path.home()
HERE = Path(__file__).resolve().parent
VENVS = {
    "whisper": HOME / "ament_ws/.venv/bin/python",
    "parakeet": HOME / "stt_parakeet_venv/bin/python",
}
SENTENCES = HOME / "projects/social-robot-speech-gaze-pipeline/stt_testset/sentences.txt"


class Worker:
    def __init__(self, spec):
        engine, _, model = spec.partition(":")
        self.name = spec
        cmd = [str(VENVS[engine]), str(HERE / "stt_worker.py"), "--engine", engine]
        if model:
            cmd += ["--model", model]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def wait_ready(self):
        for line in self.proc.stdout:
            if '"ready"' in line:
                return True
        return False

    def ask(self, wav_path, out):
        self.proc.stdin.write(str(wav_path) + "\n")
        self.proc.stdin.flush()
        out[self.name] = json.loads(self.proc.stdout.readline())

    def close(self):
        self.proc.stdin.close()
        self.proc.terminate()


def save_wav(path, audio):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def keyboard_switch(workers, state):
    """Background thread: type a number + Enter to change the active model."""
    for line in sys.stdin:
        key = line.strip()
        if key.isdigit() and 1 <= int(key) <= len(workers):
            state["active"] = int(key) - 1
            print(f"\n  >>> active model is now {key}: {workers[state['active']].name}\n", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="parakeet or whisper:<model>, e.g. whisper:large-v3-turbo")
    parser.add_argument("--mode", choices=["switch", "side"], default="switch")
    parser.add_argument("--place", default="Home")
    args = parser.parse_args()

    sentences = [s.strip() for s in SENTENCES.read_text().splitlines() if s.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = HERE / "results" / f"live_{args.mode}_{stamp}"
    out_dir.mkdir(parents=True)

    print("Loading models (each in its own venv) ...", flush=True)
    workers = [Worker(spec) for spec in args.models]
    for w in workers:
        print(f"  {w.name:45} {'ready' if w.wait_ready() else 'FAILED to start'}", flush=True)

    vad = get_vad_model()
    state = {"active": 0}
    if args.mode == "switch":
        print("\nSwitch mode: type the number + Enter to change the model at any time:")
        for i, w in enumerate(workers, 1):
            print(f"  {i} = {w.name}")
        threading.Thread(target=keyboard_switch, args=(workers, state), daemon=True).start()

    log_file = open(out_dir / "live_compare.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["turn", "place", "mode", "reference", "speech_s", "model", "text", "seconds"])
    rows = []

    turn = 0
    try:
        while True:
            reference = sentences[turn % len(sentences)]
            note = f"   [model: {workers[state['active']].name}]" if args.mode == "switch" else ""
            print(f"\n[{turn + 1}] Please say:  \"{reference}\"   (Ctrl+C to finish){note}", flush=True)
            mic = start_mic()
            audio, _ = record_sentence(mic, vad)
            stop_mic(mic)
            if audio is None:
                print("Microphone stopped (is the ReSpeaker connected?)")
                break
            turn += 1
            wav_path = out_dir / f"turn_{turn:02d}.wav"
            save_wav(wav_path, audio)

            asked = workers if args.mode == "side" else [workers[state["active"]]]
            results = {}
            threads = [threading.Thread(target=w.ask, args=(wav_path, results)) for w in asked]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            for w in asked:
                r = results[w.name]
                print(f"  {w.name:45} {r['seconds']:6.2f} s  {r['text'] or r.get('error') or '(nothing)'}")
                log.writerow([turn, args.place, args.mode, reference, f"{len(audio) / SAMPLE_RATE:.2f}",
                              w.name, r["text"], r["seconds"]])
                rows.append((w.name, reference, r["text"], r["seconds"]))
            log_file.flush()
    except KeyboardInterrupt:
        print("\nFinished.")
    finally:
        log_file.close()
        for w in workers:
            w.close()

    # Summary: WER per model (Whisper normaliser + jiwer, as in the recorded tests).
    if rows:
        import jiwer
        from whisper_normalizer.english import EnglishTextNormalizer
        normalize = EnglishTextNormalizer()
        print(f"\nSummary ({turn} sentences, place {args.place}, mode {args.mode}):")
        for w in workers:
            mine = [r for r in rows if r[0] == w.name]
            if not mine:
                continue
            wer = 100 * jiwer.wer([normalize(r[1]) for r in mine], [normalize(r[2]) or "-" for r in mine])
            times = sorted(r[3] for r in mine)
            print(f"  {w.name:45} WER {wer:5.1f} %   median time {times[len(times) // 2]:.2f} s   ({len(mine)} sentences)")
        print(f"Log and audio: {out_dir}")


if __name__ == "__main__":
    main()

