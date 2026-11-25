"""Save Silero speech segments per clip (same settings as round1), so other engines use identical cuts."""

import json

import round1_faster_whisper as R1
from faster_whisper.vad import get_speech_timestamps

segments = {name: get_speech_timestamps(audio, R1.SILERO) for name, _, audio, _ in R1.load_testset()}
out = R1.RESULTS / "silero_segments.json"
out.write_text(json.dumps(segments))
print(f"{len(segments)} clips -> {out}")
