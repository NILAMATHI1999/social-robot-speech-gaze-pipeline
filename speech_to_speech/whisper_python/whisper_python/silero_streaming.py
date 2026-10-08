"""
Reusable speech-segmentation logic for streaming Silero VAD.

This module does not contain ROS code. It receives audio frames and
Silero speech probabilities, then returns a complete speech segment.
"""

from collections import deque # Used for the preroll buffer.

import numpy as np


import os

from faster_whisper.utils import get_assets_path # You are not separately downloading your own Silero model path. The code takes the Silero ONNX model bundled with Faster-Whisper:
# Faster-Whisper installation        ↓ assets folder       ↓ silero_vad_v6.onnx
class SileroOnnxStream: # Take one audio frame and return the probability that it contains speech.
    """Run Faster-Whisper's bundled Silero ONNX model frame by frame."""

    FRAME_SIZE = 512
    CONTEXT_SIZE = 64

    def __init__(self, model_path=None):
        import onnxruntime

        if model_path is None:
            model_path = os.path.join(
                get_assets_path(),
                'silero_vad_v6.onnx',
            )

        # Use one CPU thread to keep streaming latency predictable. For a tiny streaming VAD model, you don't necessarily need many CPU threads.
        session_options = onnxruntime.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1
        session_options.enable_cpu_mem_arena = False
        session_options.log_severity_level = 4

        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
            sess_options=session_options,
        )

        self.reset()

    def reset(self):
        """Clear the neural model state before a new audio stream."""
# * Resets the internal state of the Silero neural VAD. * Clears the `hidden_state`, `cell_state`, and previous audio `context`. * Allows Silero to start a new audio stream with a clean state. * Prevents previous audio information from affecting the next stream. * Maintains short-term continuity between consecutive 32 ms audio frames. 
        self.hidden_state = np.zeros(
            (1, 1, 128),
            dtype=np.float32,
        )

        self.cell_state = np.zeros(
            (1, 1, 128),
            dtype=np.float32,
        )

        self.context = np.zeros(
            self.CONTEXT_SIZE,
            dtype=np.float32,
        )
# 64 samples of context are also retained. So Silero isn't receiving only: current 512 + 64 previous samples

    def speech_probability(self, frame):
        """Return the probability that one frame contains speech."""
# input : 512 float32 audio samples, output : probability
        frame = np.asarray(
            frame,
            dtype=np.float32,
        ).reshape(-1) # The neural model expects float32 audio.

# Silero cannot receive arbitrary chunks in this implementation. not 100,700 2048 samples, it must receieve 512 samples. That's why the ROS node has an audio_buffer later.
        if frame.size != self.FRAME_SIZE:
            raise ValueError(
                f'Silero requires exactly '
                f'{self.FRAME_SIZE} samples per frame.'
            )

        # Silero needs the final 64 samples from the previous frame.
        model_input = np.concatenate(
            [
                self.context,
                frame,
            ]
        ).reshape(1, -1) # previous + current samples(64 + 512 ) = 576 sample model input

        outputs = self.session.run(
            None,
            {
                'input': model_input,
                'h': self.hidden_state,
                'c': self.cell_state,
            },
        )

        speech_probabilities = outputs[0]
        self.hidden_state = outputs[1]
        self.cell_state = outputs[2]
# This gives continuity between frames. Frame 1   ↓ Silero state   ↓ Frame 2   ↓ updated state   ↓ Frame 3

        # Save context for the following frame.
        self.context = frame[
            -self.CONTEXT_SIZE:
        ].copy() # Takes the final 64 samples of this frame. Those become context for the next frame.

        return float(
            speech_probabilities.reshape(-1)[0]
        )



# SileroOnnxStream says: “This 32-ms frame looks 82% like speech.” It does not say: “The user's full sentence is finished.” That's where the next class comes in.

# This class converts lots of frame-level decisions into: ONE COMPLETE UTTERANCE
class SpeechSegmentCollector:
    """Collect audio frames classified as human speech."""

    def __init__(

        self, 
        threshold=0.5,
        silence_frames_to_end=16,
        minimum_speech_frames=2,
        preroll_frames=6,
        maximum_segment_frames=None,

# These are the default frame-based values used by SpeechSegmentCollector. They are used only if no values are provided when the class is created.  In the ROS node, these defaults are normally overridden by the values calculated from the ROS parameters in silero_streaming_node.py. Therefore:
# silero_streaming_node.py = user/ROS configuration in milliseconds
# silero_streaming.py      = actual segmentation logic using frame counts

      ):
        self.threshold = threshold
        self.silence_frames_to_end = silence_frames_to_end
        self.minimum_speech_frames = minimum_speech_frames
        self.maximum_segment_frames = maximum_segment_frames


        # Keeps audio immediately before speech begins.
        self.preroll = deque(maxlen=preroll_frames)

        self.speech_active = False
        self.segment_frames = []
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.active_frame_count = 0


    def process(
        self,
        frame,
        speech_probability,
    ):
        """
        Process one audio frame.
    
        Returns:
            None while no complete utterance is available.
    
            A NumPy array when a complete speech segment has ended.
        """
    
        frame = np.asarray(
            frame,
            dtype=np.float32,
        ).reshape(-1)
    
        is_speech = (
            speech_probability >= self.threshold
        )
    
        # Wait for speech to begin.
        if not self.speech_active: # Currently no utterance
            if not is_speech:
                self.preroll.append(frame.copy())
                return None
    
            # Speech has started.
            self.speech_active = True
            self.segment_frames = list(self.preroll)
            self.segment_frames.append(frame.copy())
    
            self.preroll.clear()
            self.speech_frame_count = 1
            self.silence_frame_count = 0
            self.active_frame_count = 1
    
            return None
    
        # Speech is currently active.
        self.segment_frames.append(frame.copy()) # Every frame is stored. Both speech and silence frames are temporarily retained.
        self.active_frame_count += 1
    
        if is_speech:
            self.speech_frame_count += 1
            self.silence_frame_count = 0
        else:
            self.silence_frame_count += 1
    
        reached_silence_endpoint = (
            self.silence_frame_count
            >= self.silence_frames_to_end
        )
    
        reached_maximum_duration = (
            self.maximum_segment_frames is not None
            and self.active_frame_count
            >= self.maximum_segment_frames
        )
    
        # Continue until silence or the safety limit ends the segment.
        if not (
            reached_silence_endpoint
            or reached_maximum_duration
        ):
            return None
    
        # Save frames before resetting the collector.
        finished_frames = self.segment_frames
    
        valid_speech = (
            self.speech_frame_count
            >= self.minimum_speech_frames
        ) # Suppose noise triggers Silero momentarily. Collector can reject it.
    
        self._reset_segment()
    
        # Reject very short noise activations. 
        if not valid_speech:
            return None
    
        return np.concatenate(
            finished_frames
        ).astype(
            np.float32,
            copy=False,
        )

    def _reset_segment(self):
        """Reset the current utterance state.""" # ready for next utterance.

        self.speech_active = False
        self.segment_frames = []
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.active_frame_count = 0
