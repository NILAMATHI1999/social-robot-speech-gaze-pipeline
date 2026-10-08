import argparse
import pathlib
import numpy as np
import cv2
import time

import threading
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from PIL import Image, ImageOps

from face_detection import RetinaFace

from l2cs import select_device, draw_gaze, getArch, Pipeline, render

CWD = pathlib.Path.cwd()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="cpu", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

class LatestFrameCamera:
    """Continuously capture and keep only the newest webcam frame."""

    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        self.lock = threading.Lock()
        self.frame = None
        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
        
if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    cam = args.cam_id
    # snapshot_path = args.snapshot
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    weights = pathlib.Path(args.snapshot).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    gaze_pipeline = Pipeline(
        weights=weights,
        arch=args.arch,
        device=device,
    )

    print("Gaze model device:", next(gaze_pipeline.model.parameters()).device)
    print("Model weights:", weights)

    camera = LatestFrameCamera(cam)
    last_results = None
    frame_count = 0
    INFERENCE_EVERY_N_FRAMES = 3

    with torch.no_grad():
        while True:
            frame = camera.read()

            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            if (
                frame_count % INFERENCE_EVERY_N_FRAMES == 0
                or last_results is None
            ):
                last_results = gaze_pipeline.step(frame)

            display_frame = frame

            if last_results is not None:
                display_frame = render(
                    display_frame,
                    last_results,
                )

            cv2.imshow("Demo", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()

    
