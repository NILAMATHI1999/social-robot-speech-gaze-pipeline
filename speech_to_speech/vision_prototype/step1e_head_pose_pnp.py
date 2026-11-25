"""Step 1E: head direction from YOLO pose keypoints with cv2.solvePnP.

Lightweight alternative to L2CS gaze: no face model, only the YOLO keypoints
(nose, eyes, ears) that Step 1 already computes. Gives head yaw/pitch in
degrees and says whether the head points at the robot (position-aware, like
the L2CS check). Head direction is NOT eye gaze.

Keys (click the window first): 1 dark, 2 room light, 3 phone torches,
4 side light, q quit.

Run: python step1e_head_pose_pnp.py --device cpu
"""

import argparse
import csv
import math
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
MODEL_PATH = "yolo26n-pose.pt"
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
HORIZONTAL_FOV_DEG = 70.0
VERTICAL_FOV_DEG = 50.0

KEYPOINT_CONF_MIN = 0.5             # same as Step 1
MIN_FACE_BRIGHTNESS = 20.0          # same as the L2CS face-quality rule
HEAD_YAW_TOLERANCE_DEG = 20.0       # OUR STARTING VALUE
HEAD_PITCH_TOLERANCE_DEG = 20.0     # OUR STARTING VALUE
HEAD_YAW_SIGN = 1.0                 # calibrate: flip to -1.0 if left/right is mirrored
HEAD_PITCH_SIGN = 1.0
SMOOTHING_FRAMES = 5                # median over the last frames per person

LOG_TO_CSV = True
CONDITION_KEYS = {ord("1"): "dark", ord("2"): "room_light",
                  ord("3"): "torches", ord("4"): "side_light"}

# COCO keypoint indexes used by YOLO pose.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4

# Generic 3D head model in mm. Camera-style axes: x = image right,
# y = down, z = away from the camera; nose tip at the origin.
# "Left" is the person's left, which appears on the image right.
HEAD_MODEL_MM = {
    NOSE: (0.0, 0.0, 0.0),
    LEFT_EYE: (32.0, -35.0, 25.0),
    RIGHT_EYE: (-32.0, -35.0, 25.0),
    LEFT_EAR: (75.0, -20.0, 100.0),
    RIGHT_EAR: (-75.0, -20.0, 100.0),
}


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def camera_matrix(width, height):
    fx = width / 2.0 / math.tan(math.radians(HORIZONTAL_FOV_DEG / 2.0))
    fy = height / 2.0 / math.tan(math.radians(VERTICAL_FOV_DEG / 2.0))
    return np.array([[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]], dtype=np.float64)


def camera_angles(width, height, x, y):
    """Horizontal and vertical angle of an image point seen from the camera."""
    K = camera_matrix(width, height)
    return (math.degrees(math.atan2(x - K[0, 2], K[0, 0])),
            math.degrees(math.atan2(K[1, 2] - y, K[1, 1])))


def head_pose(points_2d, K):
    """Return (yaw, pitch) in degrees, or None.

    points_2d: {keypoint index: (x, y)} with at least nose + both eyes + one ear.
    yaw > 0: head turned toward the image right. pitch > 0: head tilted up.
    """
    ids = sorted(points_2d)
    obj = np.array([HEAD_MODEL_MM[i] for i in ids], dtype=np.float64)
    img = np.array([points_2d[i] for i in ids], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # Direction the face points to, in camera axes (model's -z = out of the face).
    forward = R @ np.array([0.0, 0.0, -1.0])
    yaw = math.degrees(math.atan2(forward[0], -forward[2]))
    pitch = math.degrees(math.atan2(-forward[1], math.hypot(forward[0], forward[2])))
    return yaw * HEAD_YAW_SIGN, pitch * HEAD_PITCH_SIGN


def face_brightness(gray, pts):
    """Mean grey value in a box around the face keypoints."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = max(10.0, (max(xs) - min(xs)) * 0.5)
    x1, x2 = int(max(0, min(xs) - pad)), int(min(gray.shape[1], max(xs) + pad))
    y1, y2 = int(max(0, min(ys) - pad)), int(min(gray.shape[0], max(ys) + pad))
    crop = gray[y1:y2, x1:x2]
    return float(crop.mean()) if crop.size else None


# ----------------------------------------------------------------------------
# Per-frame processing
# ----------------------------------------------------------------------------
def process_frame(model, frame, histories, device):
    """Return one dict per tracked person."""
    h, w = frame.shape[:2]
    K = camera_matrix(w, h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                         device=device, verbose=False)[0]
    people = []
    if result.keypoints is None or result.boxes is None or result.boxes.id is None:
        return people

    xy_all = result.keypoints.xy.cpu().numpy()
    conf_all = result.keypoints.conf.cpu().numpy()
    ids = result.boxes.id.int().cpu().tolist()

    for track_id, xy, conf in zip(ids, xy_all, conf_all):
        pts = {i: tuple(xy[i]) for i in HEAD_MODEL_MM if conf[i] >= KEYPOINT_CONF_MIN}
        person = {"id": track_id, "points": pts, "yaw": None, "pitch": None,
                  "face_brightness": None, "state": "unknown", "reason": ""}
        people.append(person)

        has_ear = LEFT_EAR in pts or RIGHT_EAR in pts
        if not (NOSE in pts and LEFT_EYE in pts and RIGHT_EYE in pts and has_ear):
            person["reason"] = "too_few_points"
            continue

        person["face_brightness"] = face_brightness(gray, list(pts.values()))
        if person["face_brightness"] is None or person["face_brightness"] < MIN_FACE_BRIGHTNESS:
            person["reason"] = "too_dark"
            continue

        pose = head_pose(pts, K)
        if pose is None:
            person["reason"] = "pnp_failed"
            continue
        if abs(pose[0]) > 90.0:
            person["reason"] = "pose_flip"
            continue

        hist = histories.setdefault(track_id, deque(maxlen=SMOOTHING_FRAMES))
        hist.append(pose)
        yaw = float(np.median([p[0] for p in hist]))
        pitch = float(np.median([p[1] for p in hist]))
        person["yaw"], person["pitch"] = yaw, pitch

        # Position-aware: a person at the image edge must turn toward the
        # centre to face the camera, so compare with the expected angle.
        cam_yaw, cam_pitch = camera_angles(w, h, *pts[NOSE])
        expected_yaw, expected_pitch = -cam_yaw, -cam_pitch
        person["yaw_error"] = yaw - expected_yaw
        person["pitch_error"] = pitch - expected_pitch
        facing = (abs(person["yaw_error"]) <= HEAD_YAW_TOLERANCE_DEG
                  and abs(person["pitch_error"]) <= HEAD_PITCH_TOLERANCE_DEG)
        person["state"] = "yes" if facing else "no"
    return people


def draw(frame, people, condition, fps):
    colors = {"yes": (0, 255, 0), "no": (0, 0, 255), "unknown": (0, 165, 255)}
    for p in people:
        color = colors[p["state"]]
        for x, y in p["points"].values():
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        if NOSE in p["points"]:
            nx, ny = p["points"][NOSE]
            if p["yaw"] is not None:
                # Arrow in the head direction.
                ex = nx + 80 * math.sin(math.radians(p["yaw"]))
                ey = ny - 80 * math.sin(math.radians(p["pitch"]))
                cv2.arrowedLine(frame, (int(nx), int(ny)), (int(ex), int(ey)), color, 2)
                text = f"id{p['id']} {p['state']} yaw {p['yaw']:.0f} pitch {p['pitch']:.0f}"
            else:
                text = f"id{p['id']} unknown {p['reason']}"
            cv2.putText(frame, text, (int(nx) - 60, int(ny) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"[{condition}] {fps:.1f} FPS  keys 1-4, q",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu or 0 (first CUDA GPU)")
    parser.add_argument("--cam", type=int, default=CAMERA_INDEX)
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    writer = None
    if LOG_TO_CSV:
        name = datetime.now().strftime("step1e_head_pose_%Y%m%d_%H%M%S.csv")
        writer = csv.writer(open(name, "w", newline=""))
        writer.writerow(["t", "condition", "room_brightness", "person_id", "face_brightness",
                         "state", "reason", "yaw", "pitch", "yaw_error", "pitch_error", "fps"])
        print(f"Logging to {name}")

    condition = "unlab"
    histories = {}
    last = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            fps = 1.0 / max(1e-3, now - last)
            last = now
            room = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
            people = process_frame(model, frame, histories, args.device)

            if writer:
                for p in people or [None]:
                    writer.writerow([f"{now:.3f}", condition, f"{room:.1f}"] + (
                        ["", "", "unknown", "no_person", "", "", "", ""] if p is None else
                        [p["id"], "" if p["face_brightness"] is None else f"{p['face_brightness']:.1f}",
                         p["state"], p["reason"],
                         "" if p["yaw"] is None else f"{p['yaw']:.1f}",
                         "" if p["pitch"] is None else f"{p['pitch']:.1f}",
                         "" if p["yaw"] is None else f"{p['yaw_error']:.1f}",
                         "" if p["pitch"] is None else f"{p['pitch_error']:.1f}"]) + [f"{fps:.1f}"])

            draw(frame, people, condition, fps)
            cv2.imshow("Step 1E head pose", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in CONDITION_KEYS:
                condition = CONDITION_KEYS[key]
                print(f"Condition: {condition}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

