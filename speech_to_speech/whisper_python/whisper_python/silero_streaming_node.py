#!/usr/bin/env python3

"""
ROS 2 streaming speech-to-text node.

Pipeline:
ReSpeaker /audio
→ Silero ONNX VAD
→ speech segment
→ Faster-Whisper
→ transcription and latency topics
"""
import math
import os
import struct

import numpy as np
try:
    import usb.core
except ImportError:
    usb = None
import rclpy

import time
from datetime import datetime

from audio_common_msgs.msg import AudioData
from faster_whisper import WhisperModel
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

from whisper_python.silero_streaming import (
    SileroOnnxStream,
    SpeechSegmentCollector,
)

from whisper_python.transcription_utils import (
    pcm_s16le_to_float32,
)


from whisper_python.transcription_worker import (
      TranscriptionWorker,
)


# Its job is basically orchestration.
class SileroStreamingNode(Node):

    FRAME_SIZE = 512
    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = (
        FRAME_SIZE / SAMPLE_RATE
    ) * 1000.0

    def __init__(self):
        super().__init__('silero_streaming_transcription')

        # Faster-Whisper parameters.
        self.declare_parameter('model_size', 'base.en')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('compute_type', 'int8')
        self.declare_parameter('language', 'en')


# These are ROS 2 configuration parameters written in human-friendly units.
# Durations are given in milliseconds so they are easy to understand and tune.
# Before creating SpeechSegmentCollector, the node converts these durations into Silero frame counts using: 1 frame = 512 samples at 16 kHz = 32 ms
# Example:
# 250 ms minimum speech  -> 8 frames
# 500 ms silence         -> 16 frames
# 200 ms preroll         -> 7 frames
# 20000 ms maximum       -> 625 frames
# These converted values are then passed to SpeechSegmentCollector.
        
        # Silero speech-detection parameters.
        self.declare_parameter('vad_threshold', 0.5)

        self.declare_parameter(
            'minimum_speech_duration_ms',
            250,
        )

        self.declare_parameter(
            'minimum_silence_duration_ms',
            500,
        )

        self.declare_parameter(
            'preroll_duration_ms',
            200,
        )

        self.declare_parameter(
            'maximum_speech_duration_ms',
            20000,
        )

        # Reject extremely quiet segments.
        self.declare_parameter(
            'minimum_rms',  #  RMS check : Too quiet? reject
            0.005,
        )
        # Reject segments whose energy is too close to the current noise floor.
        self.declare_parameter('minimum_snr_db', 6.0)
        self.declare_parameter('noise_floor_ema_alpha', 0.02)
        self.declare_parameter('enable_respeaker_direction', False)

        self.declare_parameter(
            'output_dir',
            '~/ament_ws/recordings',
        )

        model_size = self.get_parameter(
            'model_size'
        ).value

        device = self.get_parameter(
            'device'
        ).value

        compute_type = self.get_parameter(
            'compute_type'
        ).value

        language = self.get_parameter(
            'language'
        ).value

        self.language = (
            None if language == 'None' else language
        )

        self.vad_threshold = float(
            self.get_parameter(
                'vad_threshold'
            ).value
        )

        minimum_speech_ms = int(
            self.get_parameter(
                'minimum_speech_duration_ms'
            ).value
        )

        minimum_silence_ms = int(
            self.get_parameter(
                'minimum_silence_duration_ms'
            ).value
        )

        preroll_ms = int(
            self.get_parameter(
                'preroll_duration_ms'
            ).value
        )

        maximum_speech_ms = int(
            self.get_parameter(
                'maximum_speech_duration_ms'
            ).value
        )

        self.minimum_rms = float(
            self.get_parameter(
                'minimum_rms'
            ).value
        )
        self.minimum_snr_db = float(
            self.get_parameter(
                'minimum_snr_db'
            ).value
        )

        self.noise_floor_ema_alpha = float(
            self.get_parameter(
                'noise_floor_ema_alpha'
            ).value
        )

        self.enable_respeaker_direction = bool(
            self.get_parameter(
                'enable_respeaker_direction'
            ).value
        )
# Start with a very quiet baseline until non-speech frames arrive. Those lines create initial state for the new features:Higher SNR is generally better because the speech is clearer compared with background noise. The code rejects segments below the configured minimum SNR.

        self.noise_floor_rms = 1e-4 # starting background-noise level for SNR filtering.

        self.respeaker_device = None
        self.direction_warning_sent = False

        self.output_dir = os.path.expanduser(
            self.get_parameter(
                'output_dir'
            ).value
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        # Convert millisecond parameters into 512-sample frames.
        minimum_speech_frames = math.ceil(
            minimum_speech_ms
            / self.FRAME_DURATION_MS
        )

        silence_frames = math.ceil(
            minimum_silence_ms
            / self.FRAME_DURATION_MS
        )

        preroll_frames = math.ceil(
            preroll_ms
            / self.FRAME_DURATION_MS
        )

        maximum_segment_frames = math.ceil(
            maximum_speech_ms
            / self.FRAME_DURATION_MS
        )

        self.get_logger().info(
            f'Loading Faster-Whisper '
            f'{model_size} on {device}...'
        )

        self.whisper_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        self.transcription_worker = TranscriptionWorker(
            transcribe=self.transcribe,
            maximum_pending_segments=1,
        )
        self.worker_result_timer = self.create_timer(
            0.05,
            self.check_worker_results, # Every: 50 ms, ROS asks: Has the background worker completed anything or failed?
        )


        self.get_logger().info(
            'Loading Silero ONNX VAD...'
        )

        self.silero = SileroOnnxStream()

        self.collector = SpeechSegmentCollector(
            threshold=self.vad_threshold,
            silence_frames_to_end=silence_frames,
            minimum_speech_frames=minimum_speech_frames,
            preroll_frames=preroll_frames,
            maximum_segment_frames=maximum_segment_frames,
        )

        # Holds incomplete audio until 512 samples are available. 
# ROS /audio messages do not necessarily arrive in chunks of exactly 512 samples. But Silero requires: exactly 512. Therefore: ROS audio chunks -> audio_buffer -> take exactly 512  -> Silero.. Anything leftover stays for the next callback.
        
        self.audio_buffer = np.empty(
            0,
            dtype=np.float32,
        )

        self.last_speech_time = None
        self.last_published_speech_state = False

        # Continuous processed channel-0 audio. This connects directly to the previous ReSpeaker node. subscribes /audio
        self.audio_subscription = self.create_subscription(
            AudioData,
            'audio',
            self.audio_callback,
            50,
        )

        self.transcription_publisher = self.create_publisher(
            String,
            'transcription',
            10,
        )

        self.inference_latency_publisher = self.create_publisher(
            Float32,
            'whisper_inference_latency',
            10,
        )

        self.speech_end_latency_publisher = self.create_publisher(
            Float32,
            'speech_end_to_text_latency',
            10,
        )

        self.speech_state_publisher = self.create_publisher(
            Bool,
            'silero_is_speeching',
            10,
        )

        self.probability_publisher = self.create_publisher(
            Float32,
            'silero_speech_probability',
            10,
        )
        self.direction_publisher = self.create_publisher(
            Float32,
            'respeaker_direction_degrees',
            10,
        )

        if self.enable_respeaker_direction:
            self.direction_timer = self.create_timer(
                0.2,
                self.poll_respeaker_direction,
            )

        self.get_logger().info(
            'Silero streaming transcription node ready.'
        )

        self.get_logger().info(
            f'VAD threshold: {self.vad_threshold}, '
            f'speech frames: {minimum_speech_frames}, '
            f'silence frames: {silence_frames}, '
            f'maximum frames: {maximum_segment_frames}'
        )
    def audio_callback(self, msg):
        """Process continuous ReSpeaker channel-0 audio."""# ReSpeaker gives something conceptually like: signed 16-bit little-endian PCM. Silero wants float32. so convert

        raw_audio = bytes(msg.data)

        if not raw_audio:
            return

        audio = pcm_s16le_to_float32(
            raw_audio
        )

        self.audio_buffer = np.concatenate(
            [
                self.audio_buffer,
                audio,
            ]
        )

        while self.audio_buffer.size >= self.FRAME_SIZE:
            frame = self.audio_buffer[
                :self.FRAME_SIZE
            ]

            self.audio_buffer = self.audio_buffer[
                self.FRAME_SIZE:
            ]

            probability = (
                self.silero.speech_probability(frame)
            )
            # Track quiet background frames without replacing Silero VAD.
            frame_rms = float(
                np.sqrt(np.mean(frame * frame))
            )

            if (
                probability < self.vad_threshold
                and not self.collector.speech_active
            ):
                alpha = self.noise_floor_ema_alpha
                self.noise_floor_rms = (
                    (1.0 - alpha) * self.noise_floor_rms
                    + alpha * frame_rms
                )

            self.probability_publisher.publish(
                Float32(data=float(probability))
            )

            if probability >= self.vad_threshold:
                self.last_speech_time = time.perf_counter()

            segment = self.collector.process(
                frame,
                speech_probability=probability,
            )

            self.publish_speech_state()

            if segment is not None:
                transcription_job = (
                    segment,
                    self.last_speech_time,
                )

                accepted = self.transcription_worker.submit(
                    transcription_job
                )



                if not accepted:
                    self.get_logger().warning(
                        'Dropped speech segment because '
                        'Whisper is busy and the pending '
                        'queue is full.'
    

                )

    def poll_respeaker_direction(self):
        """Publish the ReSpeaker microphone-array direction."""

        if usb is None:
            if not self.direction_warning_sent:
                self.get_logger().warning(
                    'pyusb is unavailable; direction disabled.'
                )
                self.direction_warning_sent = True
            return

        try:
            if self.respeaker_device is None:
                self.respeaker_device = usb.core.find(
                    idVendor=0x2886,
                    idProduct=0x0018,
                )

            if self.respeaker_device is None:
                if not self.direction_warning_sent:
                    self.get_logger().warning(
                        'ReSpeaker USB device not found; '
                        'direction disabled.'
                    )
                    self.direction_warning_sent = True
                return

            command = 0x80 | 0 | 0x40

            response = self.respeaker_device.ctrl_transfer(
                0xC0,
                0,
                command,
                21,
                8,
                100000,
            )

            direction = struct.unpack(
                'ii',
                bytes(response),
            )[0]

            self.direction_publisher.publish(
                Float32(data=float(direction))
            )

        except Exception as error:
            self.get_logger().warning(
                f'Could not read ReSpeaker direction: {error}'
            )
    def publish_speech_state(self):
        """Publish only when the Silero speech state changes."""

        current_state = self.collector.speech_active

        if current_state == self.last_published_speech_state:
            return

        self.last_published_speech_state = current_state

        self.speech_state_publisher.publish(
            Bool(data=current_state)
        )

        if current_state:
            self.get_logger().info(
                'Silero detected speech.'
            )
        else:
            self.get_logger().info(
                'Silero detected speech end.'
            )

    def transcribe(self, transcription_job):
        """Transcribe one Silero-confirmed speech segment.""" # This is executed by the background transcription worker, not directly inside the ROS audio callback.
        
        audio, speech_end_time = transcription_job

        rms = float(
            np.sqrt(np.mean(audio * audio))
        ) # This measures overall signal energy. If: RMS < 0.005 segment gets rejected

# It calculates the speech-to-noise ratio and rejects speech that is too close to the background noise
        noise_floor = max(
            self.noise_floor_rms,
            1e-6,
        )

        snr_db = 20.0 * math.log10(
            max(rms, 1e-6) / noise_floor
        )

        if snr_db < self.minimum_snr_db:
            self.get_logger().info(
                f'Rejected low-SNR segment: '
                f'RMS={rms:.4f}, '
                f'noise={noise_floor:.4f}, '
                f'SNR={snr_db:.1f} dB'
            )
            return

        if rms < self.minimum_rms:
            self.get_logger().info(
                f'Rejected quiet segment: RMS={rms:.4f}'
            )
            return

        self.get_logger().info(
            'Transcribing Silero speech segment...'
        )

        inference_started = time.perf_counter()

        segments, info = self.whisper_model.transcribe(
            audio,
            language=self.language,
            beam_size=5, # Whisper explores multiple decoding alternatives rather than only one.  gMore accuracy potential, but more computation than greedy decoding.
            condition_on_previous_text=False,
        )

        transcription = ' '.join(
            segment.text.strip()
            for segment in segments
            if segment.text
        ).strip()

        transcription_finished = time.perf_counter()

        inference_latency = (
            transcription_finished
            - inference_started
        )

        speech_end_latency = None

        if speech_end_time is not None:
            speech_end_latency = (
                transcription_finished
                - speech_end_time
            )

        self.get_logger().info(
            f'Transcription: {transcription}'
        )

        self.get_logger().info(
            f'Whisper inference time: '
            f'{inference_latency:.2f} seconds'
        )

        if speech_end_latency is not None:
            self.get_logger().info(
                f'Speech-end-to-text latency: '
                f'{speech_end_latency:.2f} seconds'
            )

        if transcription:
            self.transcription_publisher.publish(
                String(data=transcription)
            )

        self.inference_latency_publisher.publish(
            Float32(data=float(inference_latency))
        )

        if speech_end_latency is not None:
            self.speech_end_latency_publisher.publish(
                Float32(data=float(speech_end_latency))
            )

        self.save_result(
            transcription=transcription,
            inference_latency=inference_latency,
            speech_end_latency=speech_end_latency,
            detected_language=info.language,
        )


    def check_worker_results(self):
        """Remove completed worker results and report failures."""# Your worker can complete successfully or throw an exception.

        while True:
            completed = (
                self.transcription_worker.get_completed_result()
            )

            if completed is None:
                return

            succeeded, result = completed

            if not succeeded:
                self.get_logger().error(
                    f'Background transcription failed: {result}'
                )


    def save_result(
        self,
        transcription,
        inference_latency,
        speech_end_latency,
        detected_language,
    ):
        """Save one experiment result."""

        timestamp = datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )

        filename = os.path.join(
            self.output_dir,
            f'silero_transcription_{timestamp}.txt',
        )

        with open(
            filename,
            'w',
            encoding='utf-8',
        ) as output_file:
            output_file.write(
                f'Inference latency: '
                f'{inference_latency:.2f}s\n'
            )

            if speech_end_latency is not None:
                output_file.write(
                    f'Speech-end-to-text latency: '
                    f'{speech_end_latency:.2f}s\n'
                )

            output_file.write(
                f'Language: {detected_language}\n'
            )

            output_file.write(
                f'Transcription: {transcription}\n'
            )


def main(args=None):
    rclpy.init(args=args)

    node = SileroStreamingNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.transcription_worker.close()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
