import numpy as np

from whisper_python.silero_streaming import (
    SileroOnnxStream,

    SpeechSegmentCollector,
)


def make_frame(value):
    """Create one 512-sample test frame."""
    return np.full(
        512,
        value,
        dtype=np.float32,
    )


def test_noise_does_not_create_speech_segment():
    collector = SpeechSegmentCollector(
        threshold=0.5,
        silence_frames_to_end=2,
        minimum_speech_frames=2,
        preroll_frames=1,
    )

    for _ in range(10):
        result = collector.process(
            make_frame(0.0),
            speech_probability=0.1,
        )

        assert result is None


def test_speech_ends_after_required_silence():
    collector = SpeechSegmentCollector(
        threshold=0.5,
        silence_frames_to_end=2,
        minimum_speech_frames=2,
        preroll_frames=1,
    )

    # One frame of audio before speech begins.
    collector.process(
        make_frame(0.0),
        speech_probability=0.1,
    )

    # Two frames classified as speech.
    collector.process(
        make_frame(1.0),
        speech_probability=0.9,
    )

    collector.process(
        make_frame(1.0),
        speech_probability=0.8,
    )

    # The first silent frame does not finish the segment.
    result = collector.process(
        make_frame(0.0),
        speech_probability=0.1,
    )

    assert result is None

    # The second silent frame finishes the segment.
    result = collector.process(
        make_frame(0.0),
        speech_probability=0.1,
    )

    assert result is not None

    # Pre-roll + 2 speech frames + 2 ending frames.
    assert len(result) == 5 * 512


def test_short_noise_spike_is_discarded():
    collector = SpeechSegmentCollector(
        threshold=0.5,
        silence_frames_to_end=2,
        minimum_speech_frames=2,
        preroll_frames=1,
    )

    # Only one frame is classified as speech.
    collector.process(
        make_frame(1.0),
        speech_probability=0.9,
    )

    collector.process(
        make_frame(0.0),
        speech_probability=0.1,
    )

    result = collector.process(
        make_frame(0.0),
        speech_probability=0.1,
    )

    # One-frame activation is too short and must be rejected.
    assert result is None
def test_silero_model_returns_valid_probability():
      """The real ONNX model must return a probability from 0 to 1."""

      detector = SileroOnnxStream()

      silent_frame = np.zeros(
          512,
          dtype=np.float32,
      )

      probability = detector.speech_probability(
          silent_frame
      )

      assert 0.0 <= probability <= 1.0

def test_speech_segment_ends_at_maximum_duration():
    """Continuous speech must end at the configured safety limit."""

    collector = SpeechSegmentCollector(
        threshold=0.5,
        silence_frames_to_end=2,
        minimum_speech_frames=2,
        preroll_frames=1,
        maximum_segment_frames=4,
    )

    # One pre-roll frame before speech.
    collector.process(
        make_frame(0.0),
        speech_probability=0.1,
    )

    result = None

    # Continuous speech without silence.
    for _ in range(4):
        result = collector.process(
            make_frame(1.0),
            speech_probability=0.9,
        )

    assert result is not None

    # One pre-roll frame plus four active frames.
    assert len(result) == 5 * 512


