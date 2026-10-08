"""Bounded background worker for speech transcription."""

import queue
import threading


_STOP = object()


class TranscriptionWorker:
    """Run transcription without blocking the audio callback."""

    def __init__(
        self,
        transcribe,
        maximum_pending_segments=1,
    ):
        self.transcribe = transcribe

        self.pending_segments = queue.Queue(
            maxsize=maximum_pending_segments,
        )

        self.completed_results = queue.Queue()

        self.closed = False

        self.thread = threading.Thread(
            target=self._run,
            name='transcription-worker',
            daemon=True,
        )

        self.thread.start()

    def submit(self, segment):
        """
        Submit a speech segment without waiting.

        Returns False when the pending queue is already full.
        """

        if self.closed:
            return False

        try:
            self.pending_segments.put_nowait(segment)
            return True

        except queue.Full:
            return False

    def get_completed_result(self):
        """Return one completed result, or None when unavailable."""

        try:
            return self.completed_results.get_nowait()

        except queue.Empty:
            return None

    def _run(self):
        """Process submitted segments sequentially."""

        while True:
            segment = self.pending_segments.get()

            try:
                if segment is _STOP:
                    return

                try:
                    result = self.transcribe(segment)

                    self.completed_results.put(
                        (True, result)
                    )

                except Exception as error:
                    self.completed_results.put(
                        (False, error)
                    )

            finally:
                self.pending_segments.task_done()

    def close(self):
        """Finish queued work and stop the worker thread."""

        if self.closed:
            return

        self.closed = True

        self.pending_segments.put(_STOP)
        self.thread.join()
