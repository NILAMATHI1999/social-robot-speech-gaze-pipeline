import threading

import numpy as np

from whisper_python.transcription_worker import (
    TranscriptionWorker,
)


def test_worker_rejects_excess_audio_backlog():
    """Only one segment may wait while transcription is busy."""

    transcription_started = threading.Event()
    allow_transcription_to_finish = threading.Event()

    processed_segments = []

    def slow_transcription(segment):
        processed_segments.append(segment)

        transcription_started.set()

        allow_transcription_to_finish.wait(
            timeout=2.0,
        )

        return segment

    worker = TranscriptionWorker(
        transcribe=slow_transcription,
        maximum_pending_segments=1,
    )

    try:
        first_segment = np.array(
            [1.0],
            dtype=np.float32,
        )

        second_segment = np.array(
            [2.0],
            dtype=np.float32,
        )

        third_segment = np.array(
            [3.0],
            dtype=np.float32,
        )

        # First segment immediately enters the worker.
        assert worker.submit(first_segment) is True

        assert transcription_started.wait(
            timeout=1.0,
        )

        # One additional segment may wait.
        assert worker.submit(second_segment) is True

        # A third segment must be rejected, not queued indefinitely.
        assert worker.submit(third_segment) is False

    finally:
        allow_transcription_to_finish.set()
        worker.close()
        

def test_worker_returns_completed_result():
      """A completed transcription must return to the ROS side."""

      transcription_finished = threading.Event()

      def transcribe(segment):
          transcription_finished.set()
          return f'processed-{segment[0]}'

      worker = TranscriptionWorker(
          transcribe=transcribe,
          maximum_pending_segments=1,
      )

      try:
          segment = np.array(
              [7.0],
              dtype=np.float32,
          )

          assert worker.submit(segment) is True

          assert transcription_finished.wait(
              timeout=1.0,
          )

          completed = None

          # Allow the worker a short time to place the result.
          for _ in range(100):
              completed = worker.get_completed_result()

              if completed is not None:
                  break

              threading.Event().wait(0.01)

          assert completed == (
              True,
              'processed-7.0',
          )

      finally:
          worker.close()



