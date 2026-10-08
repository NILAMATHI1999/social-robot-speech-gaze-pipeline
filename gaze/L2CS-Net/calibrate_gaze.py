
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from l2cs import Pipeline


# Configurable settings
CAMERA_INDEX = 0
WEIGHTS = Path("/home/robot/L2CS-Net/models/L2CSNet_gaze360.pkl")
OUTPUT = Path(__file__).parent / "gaze_calibration.json"
COUNTDOWN_SECONDS = 3
SAMPLE_SECONDS = 5
MIN_SAMPLES = 10


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    pipeline = Pipeline(
        weights=WEIGHTS,
        arch="ResNet50",
        device=torch.device("cuda:0"),
    )

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    try:
        if not camera.isOpened():
            raise RuntimeError("Cannot open camera")

        print("Only one person should be visible.")
        print("Stand about 1.5 m away, centred, and look at the lens.")
        input("Press Enter when ready.")

        for seconds in range(COUNTDOWN_SECONDS, 0, -1):
            print(f"Starting in {seconds}...")
            time.sleep(1)

        samples = []
        deadline = time.monotonic() + SAMPLE_SECONDS
        print("Keep looking at the lens...")

        with torch.inference_mode():
            while time.monotonic() < deadline:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Cannot read camera")

                result = pipeline.step(frame)

                # Accept only frames containing exactly one face.
                if len(result.bboxes) != 1:
                    continue

                yaw = float(np.degrees(np.asarray(result.yaw).reshape(-1)[0]))
                pitch = float(np.degrees(np.asarray(result.pitch).reshape(-1)[0]))

                if np.isfinite(yaw) and np.isfinite(pitch):
                    samples.append((yaw, pitch))

        if len(samples) < MIN_SAMPLES:
            raise RuntimeError(
                f"Only {len(samples)} valid samples. "
                "Improve lighting, face the lens, and try again."
            )

        neutral = np.median(samples, axis=0)
        calibration = {
            "neutral_yaw_deg": float(neutral[0]),
            "neutral_pitch_deg": float(neutral[1]),
            "sample_count": len(samples),
            "camera_index": CAMERA_INDEX,
            "mirrored": False,
            "target": "webcam lens",
        }

        OUTPUT.write_text(json.dumps(calibration, indent=2) + "\n")
        print(json.dumps(calibration, indent=2))
        print(f"Saved: {OUTPUT}")

    finally:
        camera.release()


if __name__ == "__main__":
    main()

