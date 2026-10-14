"""Step 2B: is the person who is speaking also looking at the robot?"""

import argparse
import csv
import struct
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import usb.core

from l2cs import Pipeline, select_device

from step1d_l2cs_gaze import (
    SMOOTHING_FRAMES,
    LatestFrameCamera,
    camera_angle_deg,
    classify_gaze,
    face_quality,
    smooth_gaze,
    vertical_camera_angle_deg,
    wrap_angle,
)


# Measured on 2026-10-14 with the laptop camera and the ReSpeaker flat
# under it, USB socket to the right: camera angle = 183 - ReSpeaker angle.
# Re-measure whenever the camera or the ReSpeaker is moved.
DOA_CENTRE_DEG = 183.0
DOA_SCALE = 1.0

# Faces further than this from the sound direction are not the speaker.
MATCH_TOLERANCE_DEG = 15.0

# Real faces scored 0.997+, bedsheet patterns up to 0.89.
FACE_CONFIDENCE_THRESHOLD = 0.9

ALSA_DEVICE = "hw:1,0"
MIC_CHANNELS = 6
SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.1

# Same loudness rule as speech_qwen_respeaker.py.
NOISE_CALIBRATION_CHUNKS = 10
MIN_SPEECH_RMS = 300.0
SPEECH_RMS_FACTOR = 3.0

# Keep "speaking" true across short pauses between words.
VOICE_HOLD_SECONDS = 0.5

RESPEAKER_VENDOR_ID = 0x2886
RESPEAKER_PRODUCT_ID = 0x0018



class RespeakerDirection:
    """Poll the ReSpeaker's own direction and voice flag over USB."""

    def __init__(self):
        self.device = usb.core.find(
            idVendor=RESPEAKER_VENDOR_ID,
            idProduct=RESPEAKER_PRODUCT_ID,
        )

        if self.device is None:
            raise RuntimeError("ReSpeaker not found on USB")

        self.lock = threading.Lock()
        self.direction = None
        self.voice_flag = False
        self.running = True

        self.thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
        )
        self.thread.start()

    def _read_int(self, group, offset):
        response = self.device.ctrl_transfer(
            0xC0,
            0,
            0x80 | offset | 0x40,
            group,
            8,
            100000,
        )

        return struct.unpack("ii", bytes(response))[0]

    def _poll_loop(self):
        while self.running:
            direction = self._read_int(21, 0)
            voice_flag = bool(self._read_int(19, 32))

            with self.lock:
                self.direction = direction
                self.voice_flag = voice_flag

            time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.direction, self.voice_flag

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)




class LoudnessMonitor:
    """Read ReSpeaker channel 0 and decide whether someone is speaking."""

    def __init__(self):
        self.chunk_bytes = (
            int(SAMPLE_RATE * CHUNK_SECONDS) * MIC_CHANNELS * 2
        )

        self.process = subprocess.Popen(
            [
                "arecord",
                "-q",
                "-D", ALSA_DEVICE,
                "-f", "S16_LE",
                "-r", str(SAMPLE_RATE),
                "-c", str(MIC_CHANNELS),
                "-t", "raw",
            ],
            stdout=subprocess.PIPE,
        )

        self.lock = threading.Lock()
        self.threshold = None
        self.last_rms = 0.0
        self.last_loud_time = None
        self.running = True

        self.thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
        )
        self.thread.start()

    def _chunk_rms(self, chunk):
        samples = np.frombuffer(chunk, dtype=np.int16)
        channel0 = samples.reshape(-1, MIC_CHANNELS)[:, 0]
        channel0 = channel0.astype(np.float32)

        return float(np.sqrt(np.mean(channel0 * channel0)))

    def _read_loop(self):
        noise_levels = []

        while self.running:
            chunk = self.process.stdout.read(self.chunk_bytes)

            if len(chunk) != self.chunk_bytes:
                break

            rms = self._chunk_rms(chunk)

            # The first second sets the background noise level.
            if len(noise_levels) < NOISE_CALIBRATION_CHUNKS:
                noise_levels.append(rms)

                if len(noise_levels) == NOISE_CALIBRATION_CHUNKS:
                    with self.lock:
                        self.threshold = max(
                            MIN_SPEECH_RMS,
                            float(np.median(noise_levels))
                            * SPEECH_RMS_FACTOR,
                        )
                continue

            with self.lock:
                self.last_rms = rms

                if rms >= self.threshold:
                    self.last_loud_time = time.monotonic()

    def is_speaking(self, now):
        with self.lock:
            if self.last_loud_time is None:
                return False

            return now - self.last_loud_time <= VOICE_HOLD_SECONDS

    def status(self):
        with self.lock:
            return self.threshold, self.last_rms

    def close(self):
        self.running = False
        self.process.terminate()
        self.process.wait()




def sound_camera_angle_deg(doa_deg):
    """Convert a ReSpeaker direction into the camera's angle."""
    return wrap_angle(DOA_CENTRE_DEG - doa_deg) * DOA_SCALE


def match_speaker(face_angles, sound_angle):
    """Return (face index, angle error) of the face nearest the sound.

    The index is None when no face is within MATCH_TOLERANCE_DEG.
    """
    best_index = None
    best_error = None

    for index, angle in face_angles.items():
        error = abs(wrap_angle(angle - sound_angle))

        if best_error is None or error < best_error:
            best_index = index
            best_error = error

    if best_error is None or best_error > MATCH_TOLERANCE_DEG:
        return None, best_error

    return best_index, best_error


def decide(speaking, speaker_index, speaker_gaze):
    if not speaking:
        return "silence"

    if speaker_index is None:
        return "voice_no_face"

    if speaker_gaze == "yes":
        return "speaking_and_looking"

    if speaker_gaze == "no":
        return "speaking_not_looking"

    return "speaking_gaze_unknown"




def measure_faces(frame, gaze_pipeline, gaze_histories):
    """Return {face index: box, camera angle, gaze state, yaw, pitch}."""
    height, width = frame.shape[:2]
    results = gaze_pipeline.step(frame)

    bboxes = np.asarray(results.bboxes).reshape(-1, 4)
    scores = np.asarray(results.scores).reshape(-1)

    # L2CS "pitch" is left/right, "yaw" is up/down.
    yaw_values = np.degrees(np.asarray(results.pitch).reshape(-1))
    pitch_values = np.degrees(np.asarray(results.yaw).reshape(-1))

    for index in list(gaze_histories):
        if index >= len(bboxes):
            del gaze_histories[index]

    faces = {}

    for index, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = [int(value) for value in bbox]

        quality_state, _, _, _ = face_quality(
            frame,
            bbox,
            float(scores[index]),
        )

        angle = camera_angle_deg(width, (x1 + x2) / 2.0)

        raw_gaze = classify_gaze(
            float(yaw_values[index]),
            float(pitch_values[index]),
            angle,
            vertical_camera_angle_deg(height, (y1 + y2) / 2.0),
            quality_state,
        )

        gaze_state = smooth_gaze(
            gaze_histories.setdefault(
                index,
                deque(maxlen=SMOOTHING_FRAMES),
            ),
            raw_gaze,
        )

        faces[index] = {
            "box": (x1, y1, x2, y2),
            "angle": angle,
            "gaze": gaze_state,
            "yaw": float(yaw_values[index]),
            "pitch": float(pitch_values[index]),
        }

    return faces



def draw_frame(frame, faces, speaker_index, decision, speaking, doa,
               sound_angle):
    height = frame.shape[0]

    for index, face in faces.items():
        x1, y1, x2, y2 = face["box"]
        is_speaker = index == speaker_index

        if face["gaze"] == "yes":
            color = (0, 255, 0)
        elif face["gaze"] == "no":
            color = (0, 0, 255)
        else:
            color = (0, 165, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            4 if is_speaker else 1,
        )

        cv2.putText(
            frame,
            f"face={index} gaze={face['gaze']}"
            f" angle={face['angle']:.0f}"
            + (" SPEAKER" if is_speaker else ""),
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    cv2.putText(
        frame,
        decision,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    sound_text = (
        "" if sound_angle is None else f" sound_angle={sound_angle:.0f}"
    )

    cv2.putText(
        frame,
        f"speaking={speaking} doa={doa}{sound_text}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
