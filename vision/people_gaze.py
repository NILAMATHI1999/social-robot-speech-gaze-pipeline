# people_gaze.py - T4: tracker ID per person + gaze per person + lip crops per person
# Three loops, so the slow gaze model does not slow the tracker:
#   main loop   (~8/s): camera -> YOLO-pose + tracker (body IDs) -> RetinaFace (face boxes) -> face <-> body link
#   gaze thread (~2/s): L2CS on the newest face of every ID -> LOOK / AWAY per ID
#   crop thread (25/s): lip crop of every ID (Light-ASD needs 25 pictures per second)
# Run: ~/step1_vision_venv/bin/python people_gaze.py [video|0] [--tracker tracktrack.yaml] [--save out.mp4]
import sys, json, math, time, argparse, threading, collections, cv2, numpy as np, torch
from pathlib import Path
from ultralytics import YOLO
from l2cs import Pipeline, select_device
import wave, python_speech_features                 # audio features for Light-ASD (same as gaze_trigger.py)
import sys, json, math, time, argparse, threading, collections, csv, cv2, numpy as np, torch
LIP_MIN_PCT = 50          # lips "speaking" in >= half of the frames = this person talks (chat_demo rule)
LOOK_RECENT_S = 2.0       # candidate = looked at the robot in the last 2 s
MAX_CANDIDATES = 4        # architecture: all lookers are candidates, max 4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mvp"))
sys.path.insert(0, str(ROOT / "speech_to_speech" / "vision_prototype"))
from gaze_trigger import face_crop, load_asd      # same 112x112 lip crop + Light-ASD model as the one-person version
from step1d_l2cs_gaze import LatestFrameCamera, camera_angle_deg   # webcam: newest frame only; face angle in the picture
PARTNER_GONE_S = 5.0      # architecture: partner's face gone 5 s -> UNLOCK

WEB_IMAGE = ROOT / "mvp" / "web" / "camera.jpg"   # chat_demo's web page shows this file
last_web = 0.0

WEIGHTS = ROOT / "gaze" / "L2CS-Net" / "models" / "L2CSNet_gaze360.pkl"
OFFSET = json.loads((ROOT / "mvp" / "gaze_offset.json").read_text())   # from gaze_trigger.py --calibrate
GAZE_LIMIT_DEG = 15.0          # same limit as gaze_trigger.py
FACE_TOP = 0.4                 # the face must sit in the top 40 % of the body box
# feature log: one row per visible person per sentence (gaze, lips, timing, decision) -> data to train a model later
FEATURE_LOG = ROOT / "vision" / "tracker_logs" / f"features_{time.strftime('%Y%m%d_%H%M%S')}.csv"
feat_file = open(FEATURE_LOG, "w", newline="")
feat = csv.writer(feat_file)
feat.writerow(["time", "sentence", "sentence_s", "locked_partner", "id", "gaze", "look_age_s",
               "lips_pct", "candidates", "decision"])

ap = argparse.ArgumentParser()
ap.add_argument("source", nargs="?", default="0")
ap.add_argument("--tracker", default="tracktrack.yaml")
ap.add_argument("--save", default="", help="write the drawn video here (for checking + PPT)")
args = ap.parse_args()

# YOLO model: next to the project on the Jetson (~/nila/), in the home folder on the laptop
YOLO_PATH = next(p for p in [ROOT / "yolo26n-pose.pt", Path.home() / "yolo26n-pose.pt"] if p.exists())
yolo = YOLO(str(YOLO_PATH))
gaze = Pipeline(weights=WEIGHTS, arch="ResNet50", device=select_device("cpu", batch_size=1))
lock = threading.Lock()
state = {"frame": None, "faces": {}, "gaze": {}, "look_t": {}, "decision": ("", 0.0), "stop": False,
         "partner": None, "partner_seen": 0.0, "look_run": {}}
crops = collections.defaultdict(lambda: collections.deque(maxlen=250))   # {id: (time, crop)}, last 10 s


def link(faces, bodies):
    # faces: [face box], bodies: {id: body box} -> {id: face box}
    out = {}
    for x1, y1, x2, y2 in faces:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best = None
        for pid, (bx1, by1, bx2, by2) in bodies.items():
            if bx1 <= cx <= bx2 and by1 <= cy <= by1 + FACE_TOP * (by2 - by1):
                area = (bx2 - bx1) * (by2 - by1)
                if best is None or area < best[1]:      # smallest body that holds the face = best match
                    best = (pid, area)
        if best and best[0] not in out:
            out[best[0]] = [x1, y1, x2, y2]
    return out


def gaze_loop():
    # slow model in its own thread: newest face of every ID -> LOOK / AWAY
    while not state["stop"]:
        with lock:
            frame, faces = state["frame"], dict(state["faces"])
        if frame is None or not faces:
            time.sleep(0.05)
            continue
        ids = list(faces)
        imgs = []
        for pid in ids:
            x1, y1, x2, y2 = [max(int(v), 0) for v in faces[pid]]
            imgs.append(cv2.resize(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), (224, 224)))
        with torch.no_grad():
            pitch, yaw = gaze.predict_gaze(np.stack(imgs))
        words = {}
        for k, pid in enumerate(ids):                   # L2CS swaps the names (as in gaze_trigger.py)
            y, p = math.degrees(float(pitch[k])), math.degrees(float(yaw[k]))
            look = abs(y - OFFSET["yaw_deg"]) < GAZE_LIMIT_DEG and abs(p - OFFSET["pitch_deg"]) < GAZE_LIMIT_DEG
            words[pid] = "LOOK" if look else "AWAY"
        now = time.perf_counter()
        with lock:                                      # one lock block (a Lock cannot be taken twice)
            state["gaze"] = words
            for pid, w in words.items():
                if w == "LOOK":
                    state["look_t"][pid] = now          # candidate = looked at the robot recently
            partner = state["partner"]
            runs = state["look_run"]
            for pid in ids:
                runs[pid] = runs.get(pid, 0) + 1 if words[pid] == "LOOK" else 0
            trig = [pid for pid in ids if runs[pid] >= 3] if partner is None else []
            for pid in trig:
                runs[pid] = 0
        # lines for chat_demo, same words as gaze_trigger.py: about the partner (locked) or, if none, the first looker
        focus = partner if partner is not None else next((pid for pid in ids if words[pid] == "LOOK"), None)
        if focus in words:
            fx = (faces[focus][0] + faces[focus][2]) / 2
            print(f"{words[focus]} {camera_angle_deg(frame.shape[1], fx):.1f}", flush=True)   # e.g. "LOOK -1.3"
        else:
            print("NOFACE", flush=True)
        print("IDS " + " ".join(f"{pid}:{words[pid]}" for pid in ids), flush=True)          # extra line (chat_demo ignores it)
        # TRIGGER: someone looked 3 gaze updates in a row (~1 s at 2/s) and nobody is locked yet
        for pid in trig:
            print(f"TRIGGER {pid}", flush=True)


def crop_loop():
    # 25 times per second: one lip crop per ID from the newest frame (for the lip check later)
    while not state["stop"]:
        with lock:
            faces = dict(state["faces"])
            # live: the camera's newest frame (30/s) like gaze_trigger, not the slower tracker frame (frozen mouth = "not speaking")
            frame = cam.frame if live else state["frame"]
        if frame is not None:
            now = time.perf_counter()
            for pid, box in faces.items():
                crops[pid].append((now, face_crop(frame, box)))
        time.sleep(0.04)


def lips(asd, snap, t0, t1, wav_path):
    # same as gaze_trigger.lip_score, but for one ID's crops: % of frames Light-ASD scores "speaking" (None = too few crops)
    model, head = asd
    times = np.array([t for t, _ in snap])
    want = np.arange(t0, t1, 1 / 25)                         # Light-ASD expects 25 face pictures per second
    if len(snap) < 25 or len(want) < 25 or times[0] > t0 + 0.5:
        return None
    idx = np.clip(np.searchsorted(times, want), 0, len(snap) - 1)
    faces = np.array([snap[i][1] for i in idx])
    with wave.open(str(wav_path)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    mfcc = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
    n = min(len(faces), len(mfcc) // 4)
    a = torch.FloatTensor(mfcc[: n * 4]).unsqueeze(0)
    v = torch.FloatTensor(faces[:n]).unsqueeze(0)
    with torch.no_grad():
        out = model.forward_audio_visual_backend(model.forward_audio_frontend(a), model.forward_visual_frontend(v))
        scores = np.array(head.forward(out, labels=None))
    smooth = np.convolve(scores, np.ones(5) / 5, mode="same")
    return round(100 * float(np.mean(smooth > 0)))


def lip_requests():
    # the audio side writes "LIP <start> <end> <wav>" per sentence; we answer the lips of every candidate + the decision
    asd = load_asd()
    print("ready", flush=True)
    for line in sys.stdin:
        if line.strip() == "UNLOCK":                 # chat_demo's unlock rules (look away + silent, bye) -> free the lock
            with lock:
                state["partner"] = None
            print("UNLOCKED", flush=True)
            continue
        parts = line.split()
        if len(parts) != 4 or parts[0] != "LIP":
            continue
        t0, t1, wav = float(parts[1]), float(parts[2]), parts[3]
        now = time.perf_counter()
        with lock:
            look_t = dict(state["look_t"])
            snaps = {pid: list(crops[pid]) for pid in state["faces"]}
        partner = state["partner"]
        if partner is not None:                      # LOCKED: only the partner's lips count
            cands = [partner] if partner in snaps else []
        else:                                        # CONFIRM: all recent lookers are candidates
            cands = [pid for pid in snaps if now - look_t.get(pid, -99.0) <= LOOK_RECENT_S][:MAX_CANDIDATES]
        pct = {pid: lips(asd, snaps[pid], t0, t1, wav) for pid in cands}
        talking = [pid for pid, p in pct.items() if p is not None and p >= LIP_MIN_PCT]
        if len(talking) == 1:
            decision = f"PARTNER {talking[0]}"
            with lock:
                state["partner"], state["partner_seen"] = talking[0], now      # lock this ID
        elif talking:
            decision = "UNSURE"                      # two people talking: no partner (do not guess)
        else:
            decision = "NONE"
        print("LIPS " + " ".join(f"{pid}:{p}" for pid, p in pct.items()) + f" -> {decision}", flush=True)
        score = pct.get(talking[0]) if len(talking) == 1 else 0
        print(f"LIPSCORE {score}", flush=True)       # the answer chat_demo already understands (>= 50 = partner talks)
        with lock:
            state["decision"] = (decision, now)
        # one row per visible person (also the ones who were not candidates)
        with lock:
            words = dict(state["gaze"])
        lip_requests.n = getattr(lip_requests, "n", 0) + 1          # sentence counter
        for pid in snaps:
            feat.writerow([round(now, 3), lip_requests.n, round(t1 - t0, 2), partner, pid, words.get(pid, "NOFACE"),
                           round(now - look_t[pid], 2) if pid in look_t else "", pct.get(pid, ""), len(cands), decision])
        feat_file.flush()

live = args.source.isdigit()
cam = LatestFrameCamera(int(args.source)) if live else cv2.VideoCapture(args.source)

threading.Thread(target=gaze_loop, daemon=True).start()
threading.Thread(target=crop_loop, daemon=True).start()
threading.Thread(target=lip_requests, daemon=True).start()

out = None
while True:
    ok, frame = cam.read()
    if not ok:
        break
    t = time.perf_counter()
    r = yolo.track(frame, persist=True, tracker=args.tracker, classes=[0], verbose=False)[0]
    bodies = {} if r.boxes.id is None else dict(zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist()))
    found = gaze.detector(frame)                       # RetinaFace only (fast); gaze runs in its thread
    faces = link([b for b, _, s in found if s > 0.5], bodies) if found is not None else {}
    with lock:
        p = state["partner"]
        if p is not None and p in faces:
            state["partner_seen"] = time.perf_counter()
        elif p is not None and time.perf_counter() - state["partner_seen"] > PARTNER_GONE_S:
            state["partner"] = None
            print("UNLOCKED", flush=True)             # partner gone 5 s (architecture step 5)
    with lock:
        state["frame"], state["faces"] = frame, faces
        words = dict(state["gaze"])
    ms = (time.perf_counter() - t) * 1000

    view = frame.copy()
    for pid, box in bodies.items():
        word = words.get(pid, "...") if pid in faces else "NOFACE"
        col = {"LOOK": (0, 200, 0), "AWAY": (0, 140, 255)}.get(word, (150, 150, 150))
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(view, (x1, y1), (x2, y2), col, 3)
        if pid == state["partner"]:
            cv2.rectangle(view, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 255), 3)
            cv2.putText(view, "PARTNER", (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(view, f"ID{pid} {word} crops {len(crops[pid])}", (x1 + 5, max(y1 + 25, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        if pid in faces:
            fx1, fy1, fx2, fy2 = map(int, faces[pid])
            cv2.rectangle(view, (fx1, fy1), (fx2, fy2), col, 1)
    cv2.putText(view, f"tracker loop {ms:.0f} ms", (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    decision, when = state["decision"]
    if decision and time.perf_counter() - when < 3:
        cv2.putText(view, decision, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    if args.save:
        if out is None:
            out = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 10, (view.shape[1], view.shape[0]))
        out.write(view)
    
        # ~3 pictures per second for chat_demo's web page (write to a temp file, then swap: no half-written image)
    if time.perf_counter() - last_web > 0.3:
        tmp = WEB_IMAGE.with_suffix(".tmp.jpg")
        cv2.imwrite(str(tmp), cv2.resize(view, (480, 360)))
        tmp.replace(WEB_IMAGE)
        last_web = time.perf_counter()
    cv2.imshow("people gaze", view)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

state["stop"] = True
if out is not None:
    out.release()
cv2.destroyAllWindows()