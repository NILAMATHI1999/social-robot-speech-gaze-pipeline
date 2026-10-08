
from collections import deque

import cv2
import mediapipe as mp


CAMERA_INDEX = 0

# OUR STARTING VALUES; tune with your camera.
GAZE_HORIZONTAL_TOLERANCE = 0.15
GAZE_VERTICAL_TOLERANCE = 0.18
SMOOTHING_FRAMES = 7

# MediaPipe Face Mesh landmark indexes.
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

LEFT_EYE_TOP_BOTTOM = (159, 145)
RIGHT_EYE_TOP_BOTTOM = (386, 374)

LEFT_IRIS = range(468, 473)
RIGHT_IRIS = range(473, 478)


def point(landmarks, index, width, height):
    landmark = landmarks[index]
    return landmark.x * width, landmark.y * height


def average_point(landmarks, indexes, width, height):
    points = [
        point(landmarks, index, width, height)
        for index in indexes
    ]

    return (
        sum(item[0] for item in points) / len(points),
        sum(item[1] for item in points) / len(points),
    )


def horizontal_ratio(iris_x, corner_a_x, corner_b_x):
    left_x = min(corner_a_x, corner_b_x)
    right_x = max(corner_a_x, corner_b_x)
    width = right_x - left_x

    if width < 1.0:
        return None

    return (iris_x - left_x) / width


def vertical_ratio(iris_y, top_y, bottom_y):
    top = min(top_y, bottom_y)
    bottom = max(top_y, bottom_y)
    height = bottom - top

    if height < 1.0:
        return None

    return (iris_y - top) / height


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)

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

    print("Eye-gaze test started.")
    print("Look directly at the camera, then press 'c' to calibrate.")
    print("Press 'q' to quit.")

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

            gaze_values = []

            if result.multi_face_landmarks:
                for face in result.multi_face_landmarks:
                    landmarks = face.landmark

                    left_outer = point(
                        landmarks,
                        LEFT_EYE_CORNERS[0],
                        width,
                        height,
                    )
                    left_inner = point(
                        landmarks,
                        LEFT_EYE_CORNERS[1],
                        width,
                        height,
                    )

                    right_outer = point(
                        landmarks,
                        RIGHT_EYE_CORNERS[0],
                        width,
                        height,
                    )
                    right_inner = point(
                        landmarks,
                        RIGHT_EYE_CORNERS[1],
                        width,
                        height,
                    )

                    left_iris = average_point(
                        landmarks,
                        LEFT_IRIS,
                        width,
                        height,
                    )
                    right_iris = average_point(
                        landmarks,
                        RIGHT_IRIS,
                        width,
                        height,
                    )

                    left_horizontal = horizontal_ratio(
                        left_iris[0],
                        left_outer[0],
                        left_inner[0],
                    )
                    right_horizontal = horizontal_ratio(
                        right_iris[0],
                        right_outer[0],
                        right_inner[0],
                    )

                    left_top_bottom = (
                        point(
                            landmarks,
                            LEFT_EYE_TOP_BOTTOM[0],
                            width,
                            height,
                        ),
                        point(
                            landmarks,
                            LEFT_EYE_TOP_BOTTOM[1],
                            width,
                            height,
                        ),
                    )

                    right_top_bottom = (
                        point(
                            landmarks,
                            RIGHT_EYE_TOP_BOTTOM[0],
                            width,
                            height,
                        ),
                        point(
                            landmarks,
                            RIGHT_EYE_TOP_BOTTOM[1],
                            width,
                            height,
                        ),
                    )

                    left_vertical = vertical_ratio(
                        left_iris[1],
                        left_top_bottom[0][1],
                        left_top_bottom[1][1],
                    )
                    right_vertical = vertical_ratio(
                        right_iris[1],
                        right_top_bottom[0][1],
                        right_top_bottom[1][1],
                    )

                    if (
                        left_horizontal is None
                        or right_horizontal is None
                        or left_vertical is None
                        or right_vertical is None
                    ):
                        continue

                    horizontal = (
                        left_horizontal + right_horizontal
                    ) / 2.0

                    vertical = (
                        left_vertical + right_vertical
                    ) / 2.0

                    gaze_values.append(
                        (horizontal, vertical)
                    )

                    # Draw iris centers.
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

            if gaze_values:
                horizontal, vertical = gaze_values[0]

                horizontal_history.append(horizontal)
                vertical_history.append(vertical)

                smooth_horizontal = sum(
                    horizontal_history
                ) / len(horizontal_history)

                smooth_vertical = sum(
                    vertical_history
                ) / len(vertical_history)

                horizontal_error = abs(
                    smooth_horizontal - neutral_horizontal
                )
                vertical_error = abs(
                    smooth_vertical - neutral_vertical
                )

                gaze_toward_robot = (
                    horizontal_error
                    <= GAZE_HORIZONTAL_TOLERANCE
                    and vertical_error
                    <= GAZE_VERTICAL_TOLERANCE
                )

                label = (
                    "GAZE TOWARD ROBOT"
                    if gaze_toward_robot
                    else "GAZE AWAY"
                )

                color = (
                    (0, 255, 0)
                    if gaze_toward_robot
                    else (0, 0, 255)
                )

                cv2.putText(
                    frame,
                    f"h={smooth_horizontal:.2f} "
                    f"v={smooth_vertical:.2f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

                print(
                    f"gaze_h={smooth_horizontal:.2f} "
                    f"gaze_v={smooth_vertical:.2f} "
                    f"toward_robot={gaze_toward_robot}"
                )
            else:
                cv2.putText(
                    frame,
                    "NO FACE/IRIS",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Step 1D Eye Gaze", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c") and gaze_values:
                neutral_horizontal = smooth_horizontal
                neutral_vertical = smooth_vertical
                print(
                    "Calibrated neutral gaze: "
                    f"h={neutral_horizontal:.2f}, "
                    f"v={neutral_vertical:.2f}"
                )

            if key == ord("q"):
                break

    finally:
        camera.release()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
