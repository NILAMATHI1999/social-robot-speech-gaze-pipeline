# Which Python (venv) for which job

Venvs cannot be moved: they contain their own absolute paths. Always run them by their full path
(`<venv>/bin/python script.py`); `activate` breaks if a venv folder was moved.

## Laptop
| Job | Python | Notes |
|---|---|---|
| Echo / chat demo (`mvp/echo_demo.py`, `mvp/chat_demo.py`) | `~/mvp_venv/bin/python` | faster-whisper 1.2.1 + piper-tts 1.7.0 + pyusb 1.3.1 (added 11 Nov 2026, ReSpeaker direction over USB) |
| STT benchmark Whisper (`stt_benchmark/round1_faster_whisper.py`), `live_compare.py` | `~/ament_ws/.venv/bin/python` | faster-whisper, jiwer, whisper-normalizer |
| Parakeet (`round2_parakeet.py`, Parakeet worker) | `~/stt_parakeet_venv/bin/python` | sherpa-onnx; model in `~/stt_models/` |
| Granite / Qwen benchmark | `~/stt_granite_venv/bin/python`, `~/stt_qwen_venv/bin/python` | torch (CPU) |
| All vision: L2CS gaze (`step1d_l2cs_gaze.py`), head pose (`step1e_head_pose_pnp.py`), MediaPipe | `~/step1_vision_venv/bin/python` | torch 2.14.1, numpy 1.26.4, mediapipe, face-detection; `l2cs` installed 28 Oct 2026 with `pip install --no-deps -e gaze/L2CS-Net` (editable, linked to the project copy; pandas not needed). Weights: `gaze/L2CS-Net/models/L2CSNet_gaze360.pkl`. The old `.venv_l2cs` is no longer needed |
| Tracker with OpenVINO model (`vision/tracker_switch.py --model ~/yolo26n-pose_openvino_model/`) | `~/openvino_venv/bin/python` | openvino 2026.4.1 + openvino-telemetry 2025.2.0 only; all other packages are **read** from step1_vision_venv via `step1.pth` (link, nothing copied or changed). Model export: `YOLO('yolo26n-pose.pt').export(format='openvino', imgsz=640)` -> fixed to 640 |

## Exact package versions (laptop)
Full lists (`pip freeze`) are in [venv_requirements/](venv_requirements/), one file per venv.
Rebuild a venv with: `python3 -m venv ~/<venv> && ~/<venv>/bin/pip install -r venv_requirements/<venv>.txt`
(step1_vision_venv: install `l2cs` separately with `--no-deps -e gaze/L2CS-Net`, see above).
After every install: run `~/<venv>/bin/pip freeze > venv_requirements/<venv>.txt` again and add a line below.

| Date | Venv | Change |
|---|---|---|
| 11 Nov 2026 | mvp_venv | + pyusb 1.3.1 (dry-run: only pyusb) |
| 11 Nov 2026 | step1_vision_venv | + python_speech_features 0.6 (Light-ASD lip check; dry-run: only this) |
| 23 Nov 2026 | openvino_venv (new) | + openvino 2026.4.1, openvino-telemetry 2025.2.0 (dry-run: only these; numpy 1.26.4 unchanged); reads step1_vision_venv (.pth link) |

## Jetson (`ssh robot@192.168.100.172`)
| Job | Python | Notes |
|---|---|---|
| Whisper on GPU (echo / chat demo, benchmark) | `~/pepper_venv/bin/python` | shared venv: use only, never install into it |
| Piper TTS | `~/nila/piper_venv/bin/python` | own venv (set `PIPER_PYTHON` to it) |
| Scoring packages (jiwer, whisper-normalizer) | `PYTHONPATH=~/nila/stt/pylibs` | installed with `--target`, used with pepper_venv |
| Gaze (to check) | `~/nila/vision_venv/bin/python` | |
