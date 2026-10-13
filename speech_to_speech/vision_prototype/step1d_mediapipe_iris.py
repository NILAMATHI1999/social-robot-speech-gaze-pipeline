import argparse
import csv
import time
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp


CAMERA_INDEX = 0
SMOOTHING_FRAMES = 7

HORIZONTAL_TOLERANCE = 0.15
VERTICAL_TOLERANCE = 0.18

LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

LEFT_EYE_TOP_BOTTOM = (159, 145)
RIGHT_EYE_TOP_BOTTOM = (386, 374)

LEFT_IRIS = range(468, 473)
RIGHT_IRIS = range(473, 478)


def point(landmarks, index, width, height):
    p = landmarks[index]
    return p.x * width, p.y * height


def average_point(landmarks, indexes, width, height):
    points = [
        point(landmarks, i, width, height)
        for i in indexes
    ]
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def ratio(value, start, end):
    low = min(start, end)
    high = max(start, end)
    size = high - low

    if size < 1.0:
        return None

    return (value - low) / size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=CAMERA_INDEX)
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.cam)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    horizontal_history = deque(maxlen=SMOOTHING_FRAMES)
    vertical_history = deque(maxlen=SMOOTHING_FRAMES)

    neutral_horizontal = 0.5
    neutral_vertical = 0.5

    log_name = datetime.now().strftime(
        "mediapipe_gaze_%Y%m%d_%H%M%S.csv"
    )
    log_file = open(log_name, "w", newline="")
    writer = csv.writer(log_file)

    writer.writerow([
        "timestamp",
        "horizontal_ratio",
        "vertical_ratio",
        "gaze_state",
        "fps",
    ])

    print("MediaPipe iris test started.")
    print("Look at the camera and press 'c' to calibrate.")
    print("Press 'q' to quit.")
    print(f"CSV log: {log_name}")

    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)

            current_state = "unknown"
            smooth_horizontal = None
            smooth_vertical = None

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark

                left_outer = point(
                    landmarks, LEFT_EYE_CORNERS[0], width, height
                )
                left_inner = point(
                    landmarks, LEFT_EYE_CORNERS[1], width, height
                )
                right_outer = point(
                    landmarks, RIGHT_EYE_CORNERS[0], width, height
                )
                right_inner = point(
                    landmarks, RIGHT_EYE_CORNERS[1], width, height
                )

                left_iris = average_point(
                    landmarks, LEFT_IRIS, width, height
                )
                right_iris = average_point(
                    landmarks, RIGHT_IRIS, width, height
                )

                left_horizontal = ratio(
                    left_iris[0],
                    left_outer[0],
                    left_inner[0],
                )
                right_horizontal = ratio(
                    right_iris[0],
                    right_outer[0],
                    right_inner[0],
                )

                left_top = point(
                    landmarks,
                    LEFT_EYE_TOP_BOTTOM[0],
                    width,
                    height,
                )
                left_bottom = point(
                    landmarks,
                    LEFT_EYE_TOP_BOTTOM[1],
                    width,
                    height,
                )
                right_top = point(
                    landmarks,
                    RIGHT_EYE_TOP_BOTTOM[0],
                    width,
                    height,
                )
                right_bottom = point(
                    landmarks,
                    RIGHT_EYE_TOP_BOTTOM[1],
                    width,
                    height,
                )

                left_vertical = ratio(
                    left_iris[1],
                    left_top[1],
                    left_bottom[1],
                )
                right_vertical = ratio(
                    right_iris[1],
                    right_top[1],
                    right_bottom[1],
                )

                if None not in (
                    left_horizontal,
                    right_horizontal,
                    left_vertical,
                    right_vertical,
                ):
                    horizontal = (
                        left_horizontal + right_horizontal
                    ) / 2.0
                    vertical = (
                        left_vertical + right_vertical
                    ) / 2.0

                    horizontal_history.append(horizontal)
                    vertical_history.append(vertical)

                    smooth_horizontal = (
                        sum(horizontal_history)
                        / len(horizontal_history)
                    )
                    smooth_vertical = (
                        sum(vertical_history)
                        / len(vertical_history)
                    )

                    looking_at_robot = (
                        abs(smooth_horizontal - neutral_horizontal)
                        <= HORIZONTAL_TOLERANCE
                        and abs(smooth_vertical - neutral_vertical)
                        <= VERTICAL_TOLERANCE
                    )

                    current_state = (
                        "yes" if looking_at_robot else "no"
                    )

                    cv2.circle(
                        frame,
                        (int(left_iris[0]), int(left_iris[1])),
                        3,
                        (0, 255, 0),
                        -1,
                    )
                    cv2.circle(
                        frame,
                        (int(right_iris[0]), int(right_iris[1])),
                        3,
                        (0, 255, 0),
                        -1,
                    )

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now

            if current_state == "yes":
                color = (0, 255, 0)
                text = "GAZE TOWARD ROBOT"
            elif current_state == "no":
                color = (0, 0, 255)
                text = "GAZE AWAY"
            else:
                color = (0, 165, 255)
                text = "GAZE UNKNOWN"

            cv2.putText(
                frame,
                text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            if smooth_horizontal is not None:
                cv2.putText(
                    frame,
                    f"h={smooth_horizontal:.2f} "
                    f"v={smooth_vertical:.2f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                f"FPS={fps:.1f} | c=calibrate q=quit",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            writer.writerow([
                f"{time.monotonic():.6f}",
                "" if smooth_horizontal is None else f"{smooth_horizontal:.4f}",
                "" if smooth_vertical is None else f"{smooth_vertical:.4f}",
                current_state,
                f"{fps:.2f}",
            ])
            log_file.flush()

            cv2.imshow("MediaPipe Iris Gaze", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c") and smooth_horizontal is not None:
                neutral_horizontal = smooth_horizontal
                neutral_vertical = smooth_vertical
                print(
                    "Calibrated:",
                    neutral_horizontal,
                    neutral_vertical,
                )

            if key == ord("q"):
                break

    finally:
        camera.release()
        mesh.close()
        log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

