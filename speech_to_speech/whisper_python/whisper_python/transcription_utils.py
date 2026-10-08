import numpy as np




def pcm_s16le_to_float32(raw_audio):
      """Convert signed 16-bit PCM bytes into normalized float32 audio."""
      samples = np.frombuffer(raw_audio, dtype=np.int16)
      return samples.astype(np.float32) / 32768.0


class SpeechEndTracker:
      """Track the final speech-to-silence transition."""

      def __init__(self):
          self._speech_active = False
          self._last_speech_end = None

      def observe_vad(self, is_speeching, now):
          if is_speeching:
              self._speech_active = True
          elif self._speech_active:
              self._last_speech_end = now
              self._speech_active = False

      def latency_to(self, now):
          if self._last_speech_end is None:
              return None

          return now - self._last_speech_end



