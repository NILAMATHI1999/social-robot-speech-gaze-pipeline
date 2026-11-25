# Partner Selection for a Social Robot in Multi-Person Conversations

Master's thesis prototype (FAU Erlangen-Nürnberg, FAPS). A social robot often stands in front of several people.
This pipeline decides **who the robot is talking to** and passes **only that person's speech** to speech recognition,
so other voices in the room are ignored. It runs live on a laptop CPU and on an NVIDIA Jetson Orin NX.

## How it works

```
camera ──> person detection + tracking (YOLO26n-pose + TrackTrack) ──> one ID per person
       ──> face per ID ──> gaze per ID (RetinaFace + L2CS-Net): is this person looking at the robot?
       ──> lip crops per ID ──> active speaker detection (Light-ASD): are this person's lips speaking this audio?

microphone array (ReSpeaker, 4 mics) ──> Silero VAD (+ Smart Turn: finished or thinking pause?)
       ──> sentence ──> lip check for each candidate ──> exactly one speaker = partner, two = unsure = no partner
       ──> faster-whisper (speech-to-text) ──> LLM (answer, or ignore if not meant for the robot) ──> Piper (voice)
```

Conversation flow: **IDLE** (waits for a look) → **GREET** → **CONFIRM** (whose lips move?) → **LOCKED** (only the
partner's speech is processed; the tracker keeps the partner's ID) → **UNLOCK** (partner leaves, looks away and stays
silent, or says goodbye).

## Main results (laptop CPU, live, voice-guided tests)

| Block | Result |
|---|---|
| Lip check (active speaker detection) | partner talking 88–100 % "speaking"; a loud second voice while the partner's mouth is closed 0 % |
| Gaze | 9 / 9 test steps (looking, looking away, face covered, leaving, returning) |
| Tracker comparison | 6 trackers on two-person videos at 30 / 20 / 15 / 10 / 5 FPS: TrackTrack had the fewest ID errors at every frame rate |
| Speech-to-text | 1.6 s from end of speech to text (incl. 0.8 s end pause); hallucinations on noise removed by confidence filters; names fixed with hotwords |
| Vision-language model | SmolVLM2-500M yes/no questions (phone at ear, waving, pointing, holding), checked on new recordings |

Speech recognition models (Whisper sizes, distil-Whisper, Parakeet, Granite, Qwen) were also benchmarked for word error
rate and latency on CPU and on the Jetson GPU (`stt_benchmark/`).

## Folders

| Folder | Contents |
|---|---|
| `mvp/` | the live demo: `chat_demo.py` (full robot), `echo_demo.py` (speech in, same speech out), `gaze_trigger.py` (one-person gaze + lips), `smart_turn.py` (end-of-turn model) |
| `vision/` | `people_gaze.py` (tracker ID + gaze + lips per person, partner lock), `tracker_switch.py` (compare 6 trackers live or offline at fixed FPS) |
| `vlm/` | small vision-language model tests |
| `stt_benchmark/` | speech-to-text benchmark scripts and results |
| `speech_to_speech/` | earlier prototypes (gaze, voice direction vs. face direction) |
| `venv_requirements/` | exact package versions of every environment |

How to run each block: see [`DEMO_COMMANDS.txt`](DEMO_COMMANDS.txt). Environments: see [`VENVS.md`](VENVS.md).

## Models and tools

YOLO26n-pose (Ultralytics), TrackTrack / ByteTrack / BoT-SORT, RetinaFace, L2CS-Net, Light-ASD, Silero VAD,
Smart Turn v3.2, faster-whisper, SmolVLM2, Piper TTS, OpenAI API (gpt-4o-mini), ONNX Runtime, OpenVINO, PyTorch,
OpenCV, NVIDIA Jetson Orin NX.

## Not in this repository

Model weights, virtual environments, API keys and all recordings (videos, voices and conversation logs of test
participants) are intentionally excluded for privacy. API keys are read from environment variables.
