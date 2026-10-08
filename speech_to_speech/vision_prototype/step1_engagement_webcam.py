"""Step 1A - Who is interacting with the robot? (laptop webcam prototype)

Plain Python script, NOT a ROS node. It does not touch the speech pipeline.

What it does
  1. Opens the laptop webcam.
  2. Runs YOLO26n-pose with tracking (each person keeps the same id).
  3. Per person: approximate distance, body-facing, head-facing.
  4. Keeps an attention score per person (up when engaged, down otherwise).
  5. Locks the best person; releases after a timeout.
  6. Shows all values on screen and saves them to a CSV log for tuning.

Outputs needed by Step 2:  locked person id  +  that person's camera angle.

Install:  pip install ultralytics opencv-python
Run:      python step1a_engagement_webcam.py        (press q to quit)

Where the ideas come from
  - Attention score, "highest score wins", no switching while locked,
    release timer, 2 m interaction zone: Abdelrahman et al. 2022 (IEEE Access).
  - Head pose as a stand-in for gaze: Abdelrahman et al. 2022, Lu et al. 2024.
  - Close + facing rule: Foster et al. 2017.
  - All numbers marked "OUR STARTING VALUE" are ours and must be tuned (Step 1C).
"""

import csv
import math
import time
from datetime import datetime

import cv2
from ultralytics import YOLO


# ----------------------------------------------------------------------------
# Settings (tune these in Step 1C)
# ----------------------------------------------------------------------------

MODEL_NAME = "yolo26n-pose.pt"      # fallback if needed: "yolov8n-pose.pt"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Camera. Measure once: stand at a known distance and adjust until the
# distance shown on screen is right.
HORIZONTAL_FOV_DEG = 70.0           # OUR STARTING VALUE (typical laptop webcam)

# A body point is used only if YOLO is at least this sure about it.
KEYPOINT_MIN_CONFIDENCE = 0.50      # OUR STARTING VALUE

# Distance (approximate, from body size in pixels).
CLOSE_DISTANCE_M = 2.0              # Abdelrahman 2022: 2 m interaction zone
RELEASE_DISTANCE_M = 2.5            # OUR STARTING VALUE (wider, so no flicker)

# Average real body sizes used to turn pixels into metres.
TORSO_HEIGHT_M = 0.50               # shoulder line to hip line
SHOULDER_WIDTH_M = 0.40
EAR_WIDTH_M = 0.15

# Body facing.
BODY_RATIO_MIN = 0.50               # OUR STARTING VALUE: shoulder width / torso height
UPPER_BODY_RATIO_MIN = 1.60         # OUR STARTING VALUE: shoulder width / ear width
                                    # (used when the hips are not visible)

# Head facing: |nose - middle of ears| / ear-to-ear width.
# 0 = nose exactly centred, 0.5 = nose at one ear.
HEAD_OFFSET_MAX = 0.18              # OUR STARTING VALUE
# Normalized vertical nose displacement.
HEAD_PITCH_OFFSET_MAX = 0.30

# Attention score (in seconds).
LOCK_SECONDS = 1.5                  # score needed to lock (Foster ~1 s, Lu ~3 s)
SCORE_DECAY_RATE = 0.5              # score lost per second when not engaged
RELEASE_SECONDS = 2.0               # OUR STARTING VALUE: release timeout
FORGET_SECONDS = 5.0                # delete a person not seen for this long

LOG_TO_CSV = True


# COCO pose keypoint numbers used by Ultralytics.
NOSE = 0
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def distance_px(a, b):
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def valid_point(xy, conf, index):
    """Return the keypoint, or None if it is missing or not confident enough."""
    if index >= len(xy):
        return None
    if conf is not None and conf[index] < KEYPOINT_MIN_CONFIDENCE:
        return None
    point = xy[index]
    x, y = float(point[0]), float(point[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if x <= 0 and y <= 0:      # YOLO writes (0, 0) for a point it did not find
        return None
    return point


def focal_length_px(frame_width):
    return (frame_width / 2.0) / math.tan(math.radians(HORIZONTAL_FOV_DEG / 2.0))


def camera_angle_deg(frame_width, x_pixel):
    """Horizontal angle of a point in the image.

    0 = straight ahead, negative = left side of the image, positive = right.
    Must be calibrated against the ReSpeaker direction in Step 2.
    """
    return math.degrees(
        math.atan2(x_pixel - frame_width / 2.0, focal_length_px(frame_width))
    )


def measure_person(xy, conf, frame_width):
    """Turn one person's keypoints into the Step 1 measurements.

    Returns a dict with: distance_m, distance_from, body_ratio, body_facing,
    head_offset, head_facing, close, in_release_zone, engaged.
    """
    nose = valid_point(xy, conf, NOSE)
    left_ear = valid_point(xy, conf, LEFT_EAR)
    right_ear = valid_point(xy, conf, RIGHT_EAR)
    left_shoulder = valid_point(xy, conf, LEFT_SHOULDER)
    right_shoulder = valid_point(xy, conf, RIGHT_SHOULDER)
    left_hip = valid_point(xy, conf, LEFT_HIP)
    right_hip = valid_point(xy, conf, RIGHT_HIP)

    focal = focal_length_px(frame_width)

    shoulder_width = None
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_width = distance_px(left_shoulder, right_shoulder)

    ear_width = None
    if left_ear is not None and right_ear is not None:
        ear_width = distance_px(left_ear, right_ear)

    torso_height = None
    if (
        shoulder_width is not None
        and left_hip is not None
        and right_hip is not None
    ):
        torso_height = distance_px(
            midpoint(left_shoulder, right_shoulder),
            midpoint(left_hip, right_hip),
        )

    # --- Distance (approximate). Best measure first. -----------------------
    distance_m = None
    distance_from = "none"
    if torso_height is not None and torso_height > 1.0:
        distance_m = TORSO_HEIGHT_M * focal / torso_height
        distance_from = "torso"
    elif shoulder_width is not None and shoulder_width > 1.0:
        # Upper-body fallback (hips not visible, e.g. sitting at the laptop).
        distance_m = SHOULDER_WIDTH_M * focal / shoulder_width
        distance_from = "shoulders"
    elif ear_width is not None and ear_width > 1.0:
        distance_m = EAR_WIDTH_M * focal / ear_width
        distance_from = "ears"

    close = distance_m is not None and distance_m <= CLOSE_DISTANCE_M
    in_release_zone = distance_m is not None and distance_m <= RELEASE_DISTANCE_M

    # --- Body facing --------------------------------------------------------
    body_ratio = None
    body_facing = False
    if torso_height is not None and torso_height > 1.0:
        body_ratio = shoulder_width / torso_height
        body_facing = body_ratio >= BODY_RATIO_MIN
    elif (
        shoulder_width is not None
        and ear_width is not None
        and ear_width > 1.0
    ):
        # Upper-body fallback: shoulders compared with the head width.
        body_ratio = shoulder_width / ear_width
        body_facing = body_ratio >= UPPER_BODY_RATIO_MIN
        head_pitch_offset = None

    # --- Head facing --------------------------------------------------------
    head_offset = None
    head_pitch_offset = None

    if (
        nose is not None
        and left_ear is not None
        and right_ear is not None
        and ear_width is not None
        and ear_width > 1.0
    ):
        ear_mid_x = (
            left_ear[0] + right_ear[0]
        ) / 2.0

        ear_mid_y = (
            left_ear[1] + right_ear[1]
        ) / 2.0

        head_offset = (
            abs(float(nose[0]) - float(ear_mid_x))
            / ear_width
        )

        head_pitch_offset = (
            abs(float(nose[1]) - float(ear_mid_y))
            / ear_width
        )

    head_facing = (
        head_offset is not None
        and head_offset <= HEAD_OFFSET_MAX
    )

    head_pitch_facing = (
        head_pitch_offset is not None
        and head_pitch_offset <= HEAD_PITCH_OFFSET_MAX
    )

    head_facing = (
        head_facing
        and head_pitch_facing
    )

    return {
        "distance_m": distance_m,
        "distance_from": distance_from,
        "body_ratio": body_ratio,
        "body_facing": body_facing,
        "head_offset": head_offset,
        "head_facing": head_facing,
        "close": close,
        
        "head_pitch_offset": head_pitch_offset,
        "head_pitch_facing": head_pitch_facing,

        "in_release_zone": in_release_zone,
        # To become the partner: close AND body facing AND head facing.
        "engaged": close and body_facing and head_facing,
        # To stay the partner: the wider release zone is enough.
        "still_valid": in_release_zone and body_facing and head_facing,
    }


def update_score(old_score, engaged, delta_time):
    """Attention score: up while engaged, slowly down otherwise."""
    if engaged:
        new_score = old_score + delta_time
    else:
        new_score = old_score - SCORE_DECAY_RATE * delta_time
    return max(0.0, min(new_score, LOCK_SECONDS))

def choose_partner(people):
    """Pick who to lock among the people in this frame, or None.

    Candidates: engaged now AND score reached the lock threshold.
    Tie-break: most directly facing (smallest head offset), then closest.
    """
    candidates = [
        (pid, data)
        for pid, data in people.items()
        if data["engaged"] and data["score"] >= LOCK_SECONDS
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item[1]["head_offset"] if item[1]["head_offset"] is not None else 9.0,
            item[1]["distance_m"] if item[1]["distance_m"] is not None else 99.0,
        )
    )
    return candidates[0][0]


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def main():
    model = YOLO(MODEL_NAME)

    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not camera.isOpened():
        raise RuntimeError("Could not open laptop webcam")

    log_file = None
    log_writer = None
    if LOG_TO_CSV:
        log_name = datetime.now().strftime("step1a_log_%Y%m%d_%H%M%S.csv")
        log_file = open(log_name, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "time_s", "person_id", "camera_angle_deg", "attention_score",
            "locked", "engaged", "distance_m", "distance_from",
            "body_ratio", "body_facing", "head_offset", "head_facing",
        ])
        print(f"Logging to {log_name}")

    attention_scores = {}     # person id -> score
    last_seen = {}            # person id -> time last seen
    locked_person_id = None
    release_started = None

    start_time = time.monotonic()
    previous_time = start_time

    try:
        last_terminal_report = 0.0

        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read webcam frame")
                break

            now = time.monotonic()
            delta_time = min(now - previous_time, 0.2)
            previous_time = now
            frame_height, frame_width = frame.shape[:2]

            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml", verbose=False
            )
            result = results[0]
            current_people = {}

            if (
                result.keypoints is not None
                and result.boxes is not None
                and result.boxes.id is not None
            ):
                keypoints = result.keypoints.xy.cpu().numpy()
                keypoint_conf = (
                    result.keypoints.conf.cpu().numpy()
                    if result.keypoints.conf is not None
                    else None
                )
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for index, person_id in enumerate(track_ids):
                    person_id = int(person_id)
                    conf = keypoint_conf[index] if keypoint_conf is not None else None
                    data = measure_person(keypoints[index], conf, frame_width)

                    data["score"] = update_score(
                        attention_scores.get(person_id, 0.0),
                        data["engaged"],
                        delta_time,
                    )
                    attention_scores[person_id] = data["score"]
                    last_seen[person_id] = now

                    x1, y1, x2, y2 = boxes[index]
                    data["box"] = (x1, y1, x2, y2)
                    data["angle"] = camera_angle_deg(frame_width, (x1 + x2) / 2.0)
                    current_people[person_id] = data

            # People not in this frame: score goes down; forget them later.
            for person_id in list(attention_scores):
                if person_id in current_people:
                    continue
                attention_scores[person_id] = update_score(
                    attention_scores[person_id], False, delta_time
                )
                if (
                    now - last_seen.get(person_id, now) > FORGET_SECONDS
                    and person_id != locked_person_id
                ):
                    attention_scores.pop(person_id, None)
                    last_seen.pop(person_id, None)

            # Terminal debug output, limited to twice per second.
            # This must be inside the loop because `now` and
            # `current_people` are created for each camera frame.
            if now - last_terminal_report >= 0.5:
                if current_people:
                    for person_id, data in current_people.items():
                        print(
                            f"person={person_id} "
                            f"score={data['score']:.2f}s "
                            f"angle={data['angle']:.1f}° "
                            f"distance={fmt(data['distance_m'], 2)}m "
                            f"body={'Y' if data['body_facing'] else 'N'} "
                            f"head={'Y' if data['head_facing'] else 'N'} "
                            f"pitch={'Y' if data['head_pitch_facing'] else 'N'} "
                            f"engaged={'Y' if data['engaged'] else 'N'}"
                        )
                else:
                    print("No tracked people")

                last_terminal_report = now

            # Lock (only when nobody is locked: no switching while locked).
            if locked_person_id is None:
                best_id = choose_partner(current_people)
                if best_id is not None:
                    locked_person_id = best_id
                    release_started = None
                    print(
                        f"LOCKED partner={best_id} "
                        f"angle={current_people[best_id]['angle']:.1f} deg"
                    )

            # Release (only after the timeout).
            if locked_person_id is not None:
                locked_data = current_people.get(locked_person_id)
                if locked_data is not None and locked_data["still_valid"]:
                    release_started = None
                else:
                    if release_started is None:
                        release_started = now
                    if now - release_started >= RELEASE_SECONDS:
                        print(f"RELEASED partner={locked_person_id}")
                        attention_scores[locked_person_id] = 0.0
                        locked_person_id = None
                        release_started = None

            # Draw and log.
            for person_id, data in current_people.items():
                x1, y1, x2, y2 = (int(v) for v in data["box"])
                is_locked = person_id == locked_person_id
                color = (0, 255, 0) if is_locked else (0, 180, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                line1 = (
                    f"id={person_id} score={data['score']:.1f} "
                    f"angle={data['angle']:.0f}"
                    + (" LOCKED" if is_locked else "")
                )
                line2 = (
                    f"dist={fmt(data['distance_m'], 1)}m({data['distance_from']}) "
                    f"body={fmt(data['body_ratio'])}"
                    f"{'Y' if data['body_facing'] else 'N'} "
                    f"head={fmt(data['head_offset'])}"
                    f"{'Y' if data['head_facing'] else 'N'} "
                    f"pitch={fmt(data['head_pitch_offset'])}"
                    f"{'Y' if data['head_pitch_facing'] else 'N'}"
                )
                text_y = max(40, y1 - 28)
                cv2.putText(frame, line1, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, line2, (x1, text_y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                if log_writer is not None:
                    log_writer.writerow([
                        f"{now - start_time:.2f}", person_id,
                        f"{data['angle']:.1f}", f"{data['score']:.2f}",
                        int(is_locked), int(data["engaged"]),
                        fmt(data["distance_m"]), data["distance_from"],
                        fmt(data["body_ratio"]), int(data["body_facing"]),
                        fmt(data["head_offset"]), int(data["head_facing"]),
                    ])

            status = (
                f"partner: id {locked_person_id}"
                if locked_person_id is not None
                else "partner: none"
            )
            cv2.putText(frame, status, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Step 1A Engagement", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
