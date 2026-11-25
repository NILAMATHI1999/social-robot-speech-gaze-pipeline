# tracker_switch.py - person tracking: live switching, log, record, compare all trackers
# 1. Opens a video file / folder, or the webcam (0).
# 2. Finds people with YOLO-pose, gives each an ID with the chosen tracker.
# 3. Live: keys 1-6 switch tracker (1 ByteTrack 2 BoT-SORT 3 OC-SORT 4 Deep OC-SORT 5 FastTracker 6 TrackTrack),
#    r = record video on/off, q = quit.
# 4. --all: every tracker on the same video(s), no window -> fair comparison + summary.csv
# 5. --realtime: slow tracker misses frames, like live (drops frames).
# 6. --fps N: use N frames per second of video time (same result on any computer).
# 7. --model: .pt file or OpenVINO folder (run with ~/openvino_venv/bin/python).
# Log per frame: video, frame, time_s, tracker, fps, number_of_people, ids   -> vision/tracker_logs/
# extra_ids = unique IDs - most people at once (no ground truth yet: high = likely ID switches)
# Run: ~/step1_vision_venv/bin/python tracker_switch.py [video|folder|0] [--all] [--realtime] [--fps N] [--model M]
import os, csv, glob, time, argparse, cv2
from ultralytics import YOLO

# --- settings ---
# YOLO model: ~/nila/yolo26n-pose.pt on the Jetson, ~/yolo26n-pose.pt on the laptop
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
MODEL = str(next(p for p in [_root / "yolo26n-pose.pt", Path.home() / "yolo26n-pose.pt"] if p.exists()))
NAMES = ["bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack", "tracktrack", "botsort_reid", "tracktrack_reid"]   # key '1'..'8' -> tracker name
KEYS = {ord(str(i + 1)): name for i, name in enumerate(NAMES)}
# logs go next to this file in tracker_logs/
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracker_logs")
os.makedirs(LOGS, exist_ok=True)

# --- command line options ---
ap = argparse.ArgumentParser()
ap.add_argument("source", nargs="?", default="0")
ap.add_argument("--all", action="store_true")
ap.add_argument("--realtime", action="store_true")
ap.add_argument("--tracker", default="tracktrack")
# --fps N: use frames by video time, N per second (0 = every frame). Same result on any computer
ap.add_argument("--fps", type=float, default=0)
# --model: .pt file or OpenVINO folder (faster on Intel CPU)
ap.add_argument("--model", default=MODEL)
args = ap.parse_args()


# one run = one video (or webcam) with one start tracker; returns a summary
def run(source, tracker, show):
    # names for the log file, e.g. scenario_videos_duo2 / s01_t1
    name = "webcam" if source == 0 else os.path.splitext(os.path.basename(source))[0]
    folder = "" if source == 0 else os.path.basename(os.path.dirname(os.path.abspath(source)))
    models = {}                                    # one model per tracker = own ID memory
    # open video; its fps is needed for --realtime and for recording
    cap = cv2.VideoCapture(source)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # next video time (s) we want a frame for (for --fps); t0 = start time for the webcam
    next_t, t0 = 0.0, time.perf_counter()
    # test fps in the log name, e.g. _10fps
    tag = f"_{args.fps:g}fps" if args.fps else ""
    # per-frame log file: <folder>_<video>_<live|tracker>_<time>.csv
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = open(os.path.join(LOGS, f"{folder}_{name}_{'live' if show else tracker}{tag}_{stamp}.csv"), "w", newline="")
    out = csv.writer(log)
    out.writerow(["video", "frame", "time_s", "tracker", "fps", "number_of_people", "ids"])
    # counters: recorder, frame number, fps list, all IDs seen, most people at once, dropped frames
    rec, frame_no, fps_all, seen, most, dropped = None, 0, [], set(), 0, 0

    # --- main loop: one frame per round ---
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        # time of this frame in seconds: from the video, or the clock for the webcam
        time_s = (frame_no - 1) / video_fps if source != 0 else time.perf_counter() - t0
        # --fps: skip frames until the next wanted time
        if args.fps and source != 0:
            if time_s + 1e-6 < next_t:
                continue
            next_t += 1 / args.fps
        # new model the first time a tracker is used (keeps its own ID memory)
        if tracker not in models:
            models[tracker] = YOLO(args.model, task ="pose")
        # track people and time it (only tracking time = tracker fps)
        t = time.perf_counter()
        result = models[tracker].track(frame, persist=True, tracker=tracker + ".yaml",
                                       classes=[0], verbose=False)[0]      # class 0 = person
        dt = time.perf_counter() - t
        # IDs of people in this frame (empty if nobody)
        ids = [] if result.boxes.id is None else result.boxes.id.int().tolist()
        # one log line per frame
        out.writerow([f"{folder}/{name}", frame_no, round(time_s, 3), tracker, round(1 / dt, 1), len(ids), " ".join(map(str, ids))])
        fps_all.append(1 / dt); seen.update(ids); most = max(most, len(ids))

        # --realtime: skip the frames that passed while the tracker was busy
        if args.realtime and source != 0:          # frames that passed while busy are lost
            skip = max(0, int(dt * video_fps) - 1)
            for _ in range(skip):
                cap.grab()
            frame_no += skip; dropped += skip

        # --- live window (not in --all) ---
        if show:
            view = result.plot()
            cv2.putText(view, f"{tracker}  {1 / dt:.1f} FPS  ids {ids}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if rec is not None:
                cv2.circle(view, (view.shape[1] - 25, 25), 10, (0, 0, 255), -1)   # red dot = recording
                rec.write(view)
            cv2.imshow("tracker switch", view)
            # keys: q quit, r record on/off, 1-6 switch tracker
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if rec is None:
                    path = os.path.join(LOGS, f"{folder}_{name}_rec_{time.strftime('%H%M%S')}.mp4")
                    rec = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps,
                                          (view.shape[1], view.shape[0]))
                    print("recording", path)
                else:
                    rec.release(); rec = None; print("recording stopped")
            if key in KEYS:
                tracker = KEYS[key]; print("switched to", tracker)

    # close everything and return the summary (extra_ids high = likely ID switches)
    cap.release(); log.close()
    if rec is not None:
        rec.release()
    if show:
        cv2.destroyAllWindows()
    return {"video": f"{folder}/{name}", "tracker": tracker,"test_fps": args.fps or video_fps, "used_frames": len(fps_all), "frames": frame_no, "dropped": dropped,
            "avg_fps": round(sum(fps_all) / max(len(fps_all), 1), 1),
            "unique_ids": len(seen), "max_people": most, "extra_ids": len(seen) - most}


# --all: every tracker on every video (fair offline comparison) -> summary_*.csv
if args.all:
    videos = sorted(glob.glob(os.path.join(args.source, "*.mkv"))) if os.path.isdir(args.source) else [args.source]
    rows = []
    for v in videos:
        for tr in NAMES:
            r = run(v, tr, show=False)
            rows.append(r); print(r, flush=True)
    # summary name says the test: e.g. summary_10fps_..., summary_realtime_...
    tag = f"{args.fps:g}fps_" if args.fps else ""
    path = os.path.join(LOGS, f"summary_{tag}{'realtime_' if args.realtime else ''}{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("saved", path)
# live: one window, switch trackers with keys
else:
    run(0 if args.source == "0" else args.source, args.tracker, show=True)