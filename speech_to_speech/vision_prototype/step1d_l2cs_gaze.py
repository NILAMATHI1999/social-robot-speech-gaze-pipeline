
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import threading

from l2cs import Pipeline, draw_gaze, select_device
from collections import deque


MIN_FACE_WIDTH_PX = 60
MIN_FACE_SCORE = 0.50
MIN_FACE_BRIGHTNESS = 20.0

HORIZONTAL_FOV_DEG = 70.0
VERTICAL_FOV_DEG = 50.0

# Here yaw = left/right and pitch = up/down. In an unmirrored frame the
# webcam test gave yaw > 0 for the person's right (image left) and
# pitch > 0 for up, so a person looking at the lens has gaze angles of
# about +camera angle on both axes.
GAZE_YAW_SIGN = 0.5
GAZE_PITCH_SIGN = 1.0

GAZE_YAW_TOLERANCE_DEG = 10.0
GAZE_PITCH_TOLERANCE_DEG = 10.0

# Majority vote over the last frames of each face removes one-frame blips.
SMOOTHING_FRAMES = 5



class LatestFrameCamera:
    """Continuously capture and keep only the newest webcam frame."""

    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.condition = threading.Condition()
        self.frame = None
        self.frame_id = 0
        self.last_read_id = 0
        self.failed = False
        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            success, frame = self.cap.read()

            with self.condition:
                if success:
                    self.frame = frame
                    self.frame_id += 1
                else:
                    self.failed = True

                self.condition.notify_all()

            if not success:
                break

    def read(self):
        """Wait for a frame newer than the last one returned."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.failed
                or self.frame_id != self.last_read_id,
                timeout=5.0,
            )

            if self.failed or self.frame_id == self.last_read_id:
                return False, None

            self.last_read_id = self.frame_id

            return True, self.frame

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()



def wrap_angle(angle):
    return (angle + 180.0) % 360.0 - 180.0


def focal_length_px(width):
    return (
        width / 2.0
        / math.tan(math.radians(HORIZONTAL_FOV_DEG / 2.0))
    )


def camera_angle_deg(width, x):
    return math.degrees(
        math.atan2(
            x - width / 2.0,
            focal_length_px(width),
        )
    )

def vertical_camera_angle_deg(frame_height, y_pixel):
    focal = (
        frame_height / 2.0
        / math.tan(math.radians(VERTICAL_FOV_DEG / 2.0))
    )

    return math.degrees(
        math.atan2(
            y_pixel - frame_height / 2.0,
            focal,
        )
    )

def face_quality(frame, bbox, score):
    x1, y1, x2, y2 = [
        int(value) for value in bbox
    ]

    height, width = frame.shape[:2]

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    face_width = max(0, x2 - x1)

    if face_width < MIN_FACE_WIDTH_PX:
        return "unknown", "too_small", face_width, None

    if score < MIN_FACE_SCORE:
        return "unknown", "low_score", face_width, None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return "unknown", "face_missing", face_width, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    if brightness < MIN_FACE_BRIGHTNESS:
        return "unknown", "too_dark", face_width, brightness

    return "usable", "", face_width, brightness




def classify_gaze(
    yaw_deg,
    pitch_deg,
    camera_angle,
    vertical_camera_angle,
    quality_state,
):
    if quality_state != "usable":
        return "unknown"

    expected_yaw = GAZE_YAW_SIGN * camera_angle
    expected_pitch = GAZE_PITCH_SIGN * vertical_camera_angle

    yaw_error = abs(
        wrap_angle(yaw_deg - expected_yaw)
    )
    pitch_error = abs(
        wrap_angle(pitch_deg - expected_pitch)
    )

    if (
        yaw_error <= GAZE_YAW_TOLERANCE_DEG
        and pitch_error <= GAZE_PITCH_TOLERANCE_DEG
    ):
        return "yes"

    return "no"
def smooth_gaze(history, raw_state):
    if raw_state == "unknown":
        history.clear()
        return "unknown"

    history.append(raw_state)

    if history.count("yes") * 2 > len(history):
        return "yes"

    return "no"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument(
        "--weights",
        default="models/L2CSNet_gaze360.pkl",
    )
    parser.add_argument("--true-direction", default="unknown")
    parser.add_argument("--position", default="unknown")
    parser.add_argument("--distance-m", default="unknown")
    parser.add_argument("--person", default="unknown")
    parser.add_argument("--lighting", default="unknown")

    parser.add_argument(
        "--csv",
        default="",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False"
        )

    device = select_device(args.device, batch_size=1)

    weights = Path(args.weights)

    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights}"
        )

    print(f"Using device: {device}")
    print(f"Using weights: {weights}")

    gaze_pipeline = Pipeline(
        weights=weights,
        arch="ResNet50",
        device=device,
    )

    camera = LatestFrameCamera(args.cam)
    csv_name = args.csv

    if not csv_name:
        csv_name = datetime.now().strftime(
            "l2cs_gaze_%Y%m%d_%H%M%S.csv"
        )

    log_file = open(csv_name, "w", newline="")
    writer = csv.writer(log_file)

    writer.writerow([
        "timestamp",
        "face_index",
        "face_width_px",
        "brightness",
        "face_score",
        "camera_angle_deg",
        "vertical_camera_angle_deg",
        "gaze_yaw_deg",
        "gaze_pitch_deg",
        "gaze_state",
        "unknown_reason",
        "fps",
        "true_direction",
        "position",
        "distance_m",
        "person",
        "lighting",
        "prediction_correct",
        "gaze_state_raw",

    ])

    print("L2CS gaze test started.")
    print("Press q to quit.")
    print(f"CSV log: {csv_name}")

    previous_time = time.perf_counter()
    # Faces are keyed by detection index, not tracked across frames.
    gaze_histories = {}

    try:
        with torch.no_grad():
            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame")
                    break

                height, width = frame.shape[:2]

                results = gaze_pipeline.step(frame)

                bboxes = np.asarray(
                    getattr(results, "bboxes", [])
                ).reshape(-1, 4)

                scores = np.asarray(
                    getattr(results, "scores", [])
                ).reshape(-1)

                yaw_values = np.asarray(
                    getattr(results, "yaw", [])
                ).reshape(-1)

                pitch_values = np.asarray(
                    getattr(results, "pitch", [])
                ).reshape(-1)

                now = time.perf_counter()
                fps = 1.0 / max(
                    now - previous_time,
                    1e-6,
                )
                previous_time = now

                if len(bboxes) == 0:
                    writer.writerow([
                        f"{time.monotonic():.6f}",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "unknown",
                        "face_missing",
                        f"{fps:.2f}",
                        args.true_direction,
                        args.position,
                        args.distance_m,
                        args.person,
                        args.lighting,
                        "",
                        "unknown",

                    ])
                    cv2.putText(
                        frame,
                        "NO FACE",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                for index in list(gaze_histories):
                    if index >= len(bboxes):
                        del gaze_histories[index]


                for index, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = [
                        int(value) for value in bbox
                    ]

                    score = float(
                        scores[index]
                        if index < len(scores)
                        else 0.0
                    )

                    # L2CS names the angles the other way round: its
                    # "pitch" is left/right and its "yaw" is up/down.
                    yaw_deg = math.degrees(
                        float(pitch_values[index])
                    )
                    pitch_deg = math.degrees(
                        float(yaw_values[index])
                    )


                    quality_state, reason, face_width, brightness = (
                        face_quality(
                            frame,
                            bbox,
                            score,
                        )
                    )

                    center_x = (x1 + x2) / 2.0
                    camera_angle = camera_angle_deg(
                        width,
                        center_x,
                    )

                    vertical_angle = vertical_camera_angle_deg(
                        height,
                        (y1 + y2) / 2.0,
                    )

                    raw_gaze_state = classify_gaze(
                        yaw_deg,
                        pitch_deg,
                        camera_angle,
                        vertical_angle,
                        quality_state,
                    )

                    gaze_state = smooth_gaze(
                        gaze_histories.setdefault(
                            index,
                            deque(maxlen=SMOOTHING_FRAMES),
                        ),
                        raw_gaze_state,
                    )


                    if args.true_direction in ("yes", "no"):
                        prediction_correct = int(
                            gaze_state == args.true_direction
                        )
                    else:
                        prediction_correct = " "


                    if gaze_state == "yes":
                        color = (0, 255, 0)
                    elif gaze_state == "no":
                        color = (0, 0, 255)
                    else:
                        color = (0, 165, 255)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )
                    # draw_gaze expects L2CS's own (pitch, yaw) order.
                    draw_gaze(
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1,
                        frame,
                        (
                            float(pitch_values[index]),
                            float(yaw_values[index]),
                        ),
                        color=color,
                    )


                    label = (
                        f"face={index} "
                        f"gaze={gaze_state} "
                        f"yaw={yaw_deg:.1f} "
                        f"pitch={pitch_deg:.1f}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        color,
                        2,
                    )

                    writer.writerow([
                        f"{time.monotonic():.6f}",
                        index,
                        face_width,
                        ""
                        if brightness is None
                        else f"{brightness:.2f}",
                        f"{score:.3f}",
                        f"{camera_angle:.2f}",
                        f"{vertical_angle:.2f}",
                        f"{yaw_deg:.2f}",
                        f"{pitch_deg:.2f}",
                        gaze_state,
                        reason,
                        f"{fps:.2f}",
                        args.true_direction,
                        args.position,
                        args.distance_m,
                        args.person,
                        args.lighting,
                        prediction_correct,
                        raw_gaze_state,
                        
                    ])

                cv2.putText(
                    frame,
                    f"device={args.device} FPS={fps:.1f}",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                log_file.flush()
                cv2.imshow("L2CS Gaze", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

