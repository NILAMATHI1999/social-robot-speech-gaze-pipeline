"""Gaze trigger (MVP step 4): signals when the person in front has looked at the camera for 1 s.

Camera -> RetinaFace + L2CS gaze (same code as step1d_l2cs_gaze.py) -> looking at the camera?
-> looking for 1.0 s (all of the last 3 frames) -> prints "TRIGGER".
Rules adapted from the scene trigger of Athul Sukesh Nair (FAPS): calibrated gaze offset,
angle window, 1.0 s persistence, 3-frame unanimous buffer. One person, no tracking (MVP).

INPUT                      IN BETWEEN (every frame, about 2/s on laptop)              OUTPUT
webcam image  →  face found? → biggest face → good enough? (size, light) →  gaze angles (yaw, pitch)
                 → close to "looking at camera" (offset ± 20°)? → last 3 frames all yes?
                 → yes for 1.0 s in a row?                                    →  prints "TRIGGER"

Run (laptop, light on the face):
  ~/step1_vision_venv/bin/python gaze_trigger.py --calibrate   # once: look at the camera for 3 s
  ~/step1_vision_venv/bin/python gaze_trigger.py               # then: prints TRIGGER after 1 s of looking
"""

import argparse
import json
import math
import sys
import time
import threading
from pathlib import Path
import cv2                         # draw the face box + save the small camera image for the web page
import numpy as np
import torch
torch.set_num_threads(4)           # leave CPU cores for Whisper + Piper (gaze used almost all of them on the laptop)
import collections
from l2cs import Pipeline, select_device
import wave
import python_speech_features       # MFCC audio features for Light-ASD (installed 11 Nov 2026)
import wave                        # save the sentence audio for the lip check


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "speech_to_speech" / "vision_prototype"))
from step1d_l2cs_gaze import LatestFrameCamera, camera_angle_deg, face_quality   # same camera + face quality rules

WEIGHTS = HERE.parent / "gaze" / "L2CS-Net" / "models" / "L2CSNet_gaze360.pkl"
# Light-ASD folder: ~/nila/third_party on the Jetson, ~/projects/third_party on the laptop
LIGHT_ASD = next(p for p in [HERE.parent / "third_party" / "Light-ASD",
                             Path.home() / "projects" / "third_party" / "Light-ASD"] if p.exists())
ASD_WEIGHTS = LIGHT_ASD / "weight" / "finetuning_TalkSet.model"        # TalkSet weights: best on our home dataset
sys.path.insert(0, str(LIGHT_ASD))
from model.Model import ASD_Model
from loss import lossAV

OFFSET_FILE = HERE / "gaze_offset.json"
GAZE_THRESHOLD_DEG = 15.0          # how far off still counts as looking (Athul: 30; our looking data stayed within about 15 deg
PERSISTENCE_S = 1.0                # look this long to trigger
BUFFER_FRAMES = 3                  # all of the last frames must say "looking"
CALIBRATION_S = 3.0
WEB_IMAGE = HERE / "web" / "camera.jpg"   # latest camera image; chat_demo's web page shows it
WEB_IMAGE_EVERY_S = 0.3                   # save about 3 images per second
latest = {"frame": None, "box": None}     # last camera frame + biggest face box, filled by read_gaze()
CROP_SCALE = 0.40                          # Light-ASD crop: face box + 40 % margin (same as its Columbia_test.py)
crops = collections.deque(maxlen=250)      # (time, 112x112 gray face crop) of the last 10 s at ~25 fps, for the lip check


def read_gaze(pipeline, camera):
    """(yaw_deg, pitch_deg, face_angle_deg) of the biggest usable face, or None (no face, too dark, too small)."""
    latest["why"] = "no face box"

    ok, frame = camera.read()
    if not ok:
        return None
    result = pipeline.step(frame)
    latest["box"] = None                  # the picture now comes from web_image_loop (newest frame)

    bboxes = np.asarray(getattr(result, "bboxes", [])).reshape(-1, 4)
    if len(bboxes) == 0:
        return None
    scores = np.asarray(result.scores).reshape(-1)
    pitch = np.asarray(result.pitch).reshape(-1)
    yaw = np.asarray(result.yaw).reshape(-1)
    i = int(np.argmax(bboxes[:, 2] - bboxes[:, 0]))          # biggest face = person in front
    latest["box"] = bboxes[i]

    state, reason, width, bright = face_quality(frame, bboxes[i], float(scores[i]))
    if state != "usable":
        latest["why"] = f"{reason} (width {width} px, brightness {bright})"
        return None

    # Same as step1d_l2cs_gaze.py: L2CS returns the two angles under swapped names.
    face_x = (bboxes[i][0] + bboxes[i][2]) / 2                  # middle of the face box, in pixels
    return math.degrees(float(pitch[i])), math.degrees(float(yaw[i])), camera_angle_deg(frame.shape[1], face_x)   # 3rd: where the face is in the image (deg)

def web_image_loop(camera, stop):
    # own thread: save the newest camera frame ~10x per second (does not wait for the slow gaze model)
    while not stop.is_set():
        if camera.frame is not None:
            latest["frame"] = camera.frame
            save_web_image(latest.get("looking", False))
        time.sleep(0.1)

def face_crop(frame, box):
    # Light-ASD input: padded face crop -> 224x224 gray -> centre 112x112 (same as analyse_asd.py)
    x1, y1, x2, y2 = box
    bs = max(x2 - x1, y2 - y1) / 2                       # half the face size
    cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
    pad = int(bs * (1 + 2 * CROP_SCALE))
    img = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), "constant", constant_values=110)
    cy, cx = cy + pad, cx + pad
    face = img[int(cy - bs):int(cy + bs * (1 + 2 * CROP_SCALE)),
               int(cx - bs * (1 + CROP_SCALE)):int(cx + bs * (1 + CROP_SCALE))]
    gray = cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2GRAY)
    return gray[56:168, 56:168]


def crop_loop(camera, stop):
    # own thread: ~25 times per second, cut the partner's face out of the newest frame (uses the last face box)
    while not stop.is_set():
        frame, box = camera.frame, latest["box"]
        if frame is not None and box is not None:
            crops.append((time.perf_counter(), face_crop(frame, box)))
        time.sleep(0.04)

def load_asd():
    # Light-ASD model + its scoring head (the weights file stores both, with "model." / "lossAV." in front)
    model, head = ASD_Model(), lossAV()
    state = torch.load(ASD_WEIGHTS, map_location="cpu")
    model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
    head.load_state_dict({k[len("lossAV."):]: v for k, v in state.items() if k.startswith("lossAV.")})
    return model.eval(), head.eval()


def lip_score(asd, t0, t1, wav_path):
    # are the partner's lips making this sound? -> % of video frames scored "speaking" (None = not enough face crops)
    model, head = asd
    snap = list(crops)                                       # copy: crop_loop keeps adding while we work
    times = np.array([t for t, _ in snap])
    want = np.arange(t0, t1, 1 / 25)                         # Light-ASD expects 25 face pictures per second
    if len(snap) < 25 or len(want) < 25 or times[0] > t0 + 0.5:
        return None
    idx = np.clip(np.searchsorted(times, want), 0, len(snap) - 1)
    faces = np.array([snap[i][1] for i in idx])
    with wave.open(str(wav_path)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    mfcc = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)   # 100 per second
    n = min(len(faces), len(mfcc) // 4)                      # 4 audio steps per video frame
    a = torch.FloatTensor(mfcc[: n * 4]).unsqueeze(0)
    v = torch.FloatTensor(faces[:n]).unsqueeze(0)
    with torch.no_grad():
        out = model.forward_audio_visual_backend(model.forward_audio_frontend(a), model.forward_visual_frontend(v))
        scores = np.array(head.forward(out, labels=None))
    smooth = np.convolve(scores, np.ones(5) / 5, mode="same")   # same smoothing as the offline test
    return round(100 * float(np.mean(smooth > 0)))


def lip_requests(asd):
    # chat_demo writes "LIP <start> <end> <wav file>" to us; we answer "LIPSCORE <percent>" or "LIPSCORE none"
    for line in sys.stdin:
        parts = line.split()
        if len(parts) == 4 and parts[0] == "LIP":
            pct = lip_score(asd, float(parts[1]), float(parts[2]), parts[3])
            print(f"LIPSCORE {pct if pct is not None else 'none'}", flush=True)


def save_web_image(looking):
    # small JPEG with the face box (green = looking at the robot, orange = looking away) for the web page
    frame = latest["frame"]
    if frame is None:
        return
    img = frame.copy()
    if latest["box"] is not None:
        x1, y1, x2, y2 = [int(v) for v in latest["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0) if looking else (0, 140, 255), 3)
    img = cv2.resize(img, (480, 360))
    tmp = WEB_IMAGE.with_suffix(".tmp.jpg")
    cv2.imwrite(str(tmp), img)
    tmp.replace(WEB_IMAGE)            # swap in one step, so the page never gets a half-written image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="look at the camera; saves the gaze offset")
    parser.add_argument("--device", default="cpu", help="cpu (laptop) or cuda (Jetson)")
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()

    pipeline = Pipeline(weights=WEIGHTS, arch="ResNet50", device=select_device(args.device, batch_size=1))
    camera = LatestFrameCamera(args.cam)
    try:
        with torch.no_grad():
            if args.calibrate:
                print(f"Look at the camera for {CALIBRATION_S:.0f} s ...", file=sys.stderr, flush=True)
                samples, end = [], time.perf_counter() + CALIBRATION_S
                while time.perf_counter() < end:
                    gaze = read_gaze(pipeline, camera)
                    if gaze:
                        samples.append(gaze)
                if not samples:
                    print("No usable face (more light on the face? closer?)", file=sys.stderr)
                    return
                offset = {"yaw_deg": float(np.median([s[0] for s in samples])),
                          "pitch_deg": float(np.median([s[1] for s in samples]))}
                OFFSET_FILE.write_text(json.dumps(offset, indent=2))
                print(f"Saved {OFFSET_FILE.name}: {offset} ({len(samples)} frames)", file=sys.stderr)
                return

            offset = json.loads(OFFSET_FILE.read_text()) if OFFSET_FILE.exists() else {"yaw_deg": 0.0, "pitch_deg": 0.0}
            print("ready", flush=True)
            stop = threading.Event()
            threading.Thread(target=web_image_loop, args=(camera, stop), daemon=True).start()
            threading.Thread(target=crop_loop, args=(camera, stop), daemon=True).start()
            asd = load_asd()
            threading.Thread(target=lip_requests, args=(asd,), daemon=True).start()   # answers chat_demo's LIP questions

            buffer, looking_since, last_saved = [], None, 0.0
            while True:
                gaze = read_gaze(pipeline, camera)
                looking = (gaze is not None
                           and abs(gaze[0] - offset["yaw_deg"]) < GAZE_THRESHOLD_DEG
                           and abs(gaze[1] - offset["pitch_deg"]) < GAZE_THRESHOLD_DEG)
                                # Every frame, tell chat_demo what the camera sees (for engagement during the chat):
                # LOOK = looking at the robot, AWAY = face there but looking away, NOFACE = nobody / unusable face.
                word = "LOOK" if looking else ("AWAY" if gaze else "NOFACE")
                print(f"{word} {gaze[2]:.1f}" if gaze else word, flush=True)   # e.g. "LOOK -1.3" = looking, face 1.3 deg left of centre

                buffer = (buffer + [looking])[-BUFFER_FRAMES:]
                now = time.perf_counter()
                if all(buffer):
                    looking_since = looking_since or now
                    if now - looking_since >= PERSISTENCE_S:
                        print("TRIGGER", flush=True)
                        buffer, looking_since = [], None
                else:
                    looking_since = None
                status = f"NOFACE: {latest['why']}" if gaze is None else f"yaw {gaze[0]:6.1f} pitch {gaze[1]:6.1f}"
                print(f"looking={looking!s:5}  {status}  crops {len(crops)}", file=sys.stderr, flush=True)
                
                latest["looking"] = looking               # colour of the box in the web image

    finally:
        camera.release()


if __name__ == "__main__":
    main()
