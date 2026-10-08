import numpy as np
import pytest

from whisper_python.transcription_utils import (
    SpeechEndTracker,
    pcm_s16le_to_float32,
)


def test_pcm_audio_is_normalized():
    raw = np.array(
        [-32768, 0, 16384, 32767],
        dtype=np.int16,
    ).tobytes()

    result = pcm_s16le_to_float32(raw)

    expected = np.array(
        [-1.0, 0.0, 0.5, 32767 / 32768],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected)


def test_speech_end_latency_is_measured():
    tracker = SpeechEndTracker()

    tracker.observe_vad(True, now=10.0)
    tracker.observe_vad(False, now=12.5)

    assert tracker.latency_to(13.9) == pytest.approx(1.4)


def test_initial_false_is_not_speech_end():
    tracker = SpeechEndTracker()

    tracker.observe_vad(False, now=5.0)

    assert tracker.latency_to(6.0) is None

