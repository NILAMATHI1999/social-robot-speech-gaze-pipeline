"""Smart Turn v3.2 (Pipecat, BSD-2): has the person FINISHED speaking, or is it a thinking pause?
Same steps as pipecat-ai/smart-turn inference.py (last 8 s, zeros at the start, normalised, Whisper features),
but the features come from faster-whisper: identical to transformers' WhisperFeatureExtractor
(max difference 0.00002 in our check, 24 Nov 2026), so no new package is needed."""
import numpy as np
import onnxruntime as ort                                   # runs the ONNX model file (already in mvp_venv)
from pathlib import Path
from faster_whisper.feature_extractor import FeatureExtractor   # sound -> "voice picture" (same as Whisper's input)

# --- settings ---
# where the model file is (8.7 MB, downloaded from Hugging Face pipecat-ai/smart-turn-v3)
MODEL = Path(__file__).resolve().parent / "models" / "smart-turn-v3.2-cpu.onnx"   # Jetson later: -gpu.onnx
# 16,000 audio samples per second; the model looks at most at the LAST 8 seconds
SR, MAX_S = 16000, 8

# --- load once at start (not every sentence) ---
# voice picture maker: 80 frequency bands x 800 time steps for 8 s of audio
_fe = FeatureExtractor(feature_size=80, chunk_length=MAX_S)
# ONNX settings, the same as the official repo
_so = ort.SessionOptions()
_so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL          # same session settings as the repo
_so.inter_op_num_threads = 1
_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# the loaded model, ready to answer
_session = ort.InferenceSession(str(MODEL), sess_options=_so)


def turn_finished(audio):
    """audio: float32 at 16 kHz, the WHOLE turn so far -> probability the person has finished (> 0.5 = finished)."""
    # the end of the sentence matters most: keep only the last 8 s
    audio = audio[-MAX_S * SR:]                                       # keep the last 8 s
    # shorter than 8 s: add silence at the START, so the speech sits at the end (like in training)
    audio = np.pad(audio, (MAX_S * SR - len(audio), 0))               # shorter: zeros at the start (as in the repo)
    # make loud and quiet voices comparable (mean 0, spread 1)
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)      # do_normalize=True
    # sound -> voice picture, plus a batch dimension: shape (1, 80, 800)
    feats = _fe(audio.astype(np.float32), padding=0)[np.newaxis].astype(np.float32)   # (1, 80, 800)
    # the model answers one number 0..1 = how likely the person has finished speaking
    return float(_session.run(None, {"input_features": feats})[0][0].item())
