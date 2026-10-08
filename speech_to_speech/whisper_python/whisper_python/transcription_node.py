#!/usr/bin/env python3

"""
ROS 2 node for live transcription using Faster-Whisper.

Input:
    /speech_audio
    /is_speeching

Output:
    /transcription
    /whisper_inference_latency
    /speech_end_to_text_latency
"""

import os
import time
import traceback
from datetime import datetime

import rclpy
from audio_common_msgs.msg import AudioData
from faster_whisper import WhisperModel
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

from whisper_python.transcription_utils import (
    SpeechEndTracker,
    pcm_s16le_to_float32,
)


class WhisperTranscriptionNode(Node):

    def __init__(self):
        super().__init__('whisper_transcription_node')

        # Faster-Whisper configuration parameters.
        self.declare_parameter('model_size', 'base.en')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('output_dir', '../recordings')
        self.declare_parameter('save_audio', False)

        # CPU is the default for the laptop.
        # Use device:=cuda later on the Jetson.
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('compute_type', 'int8')
        self.declare_parameter('language', 'en')

        # Read ROS parameters.
        model_size = self.get_parameter('model_size').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.output_dir = self.get_parameter('output_dir').value
        self.save_audio = self.get_parameter('save_audio').value

        device = self.get_parameter('device').value
        compute_type = self.get_parameter('compute_type').value
        language = self.get_parameter('language').value

        # Convert the text "None" into the Python value None.
        self.language = language if language != 'None' else None

        # Tracks the final VAD true-to-false transition.
        self.speech_end_tracker = SpeechEndTracker()

        # Load Faster-Whisper.
        self.get_logger().info(
            f'Loading Faster-Whisper model '
            f'({model_size}) on {device}...'
        )

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        self.get_logger().info('Model loaded successfully.')

        # Create the output directory if it does not exist.
        os.makedirs(self.output_dir, exist_ok=True)

        # Receive completed speech audio from the ReSpeaker node.
        self.audio_subscription = self.create_subscription(
            AudioData,
            'speech_audio',
            self.audio_callback,
            10,
        )

        # Receive hardware VAD true/false transitions.
        self.vad_subscription = self.create_subscription(
            Bool,
            'is_speeching',
            self.vad_callback,
            10,
        )

        # Publish the final recognized text.
        self.transcription_publisher = self.create_publisher(
            String,
            'transcription',
            10,
        )

        # Publish Faster-Whisper processing time.
        self.inference_latency_publisher = self.create_publisher(
            Float32,
            'whisper_inference_latency',
            10,
        )

        # Publish time from speech ending until text is ready.
        self.speech_end_latency_publisher = self.create_publisher(
            Float32,
            'speech_end_to_text_latency',
            10,
        )

        self.get_logger().info(
            'Whisper transcription node initialized.'
        )

    def vad_callback(self, msg):
        """Record VAD transitions from the ReSpeaker."""

        self.speech_end_tracker.observe_vad(
            msg.data,
            now=time.perf_counter(),
        )

    def audio_callback(self, msg):
        """Transcribe one completed speech utterance."""

        self.get_logger().info(
            'Received speech audio. Transcribing...'
        )

        try:
            # Convert the ROS byte array into raw PCM bytes.
            raw_audio = bytes(msg.data)

            # Convert signed 16-bit PCM into normalized float32 audio.
            audio_float = pcm_s16le_to_float32(raw_audio)

            # Start measuring Faster-Whisper inference time.
            transcription_started = time.perf_counter()

            segments, info = self.model.transcribe(
                audio_float,
                language=self.language,
                beam_size=5,
            )

            # Iterating over segments performs the actual transcription.
            transcription = ' '.join(
                segment.text.strip()
                for segment in segments
            ).strip()

            # Record when the final transcription becomes available.
            transcription_finished = time.perf_counter()

            # Calculate Whisper inference time.
            inference_latency = (
                transcription_finished
                - transcription_started
            )

            # Calculate speech-end-to-final-text latency.
            speech_end_latency = (
                self.speech_end_tracker.latency_to(
                    transcription_finished
                )
            )

            detected_language = info.language

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
            else:
                self.get_logger().warning(
                    'Speech-end timestamp was unavailable.'
                )

            # Publish the recognized text.
            self.transcription_publisher.publish(
                String(data=transcription)
            )

            # Publish Whisper inference latency.
            self.inference_latency_publisher.publish(
                Float32(data=float(inference_latency))
            )

            # Publish complete speech-end-to-text latency.
            if speech_end_latency is not None:
                self.speech_end_latency_publisher.publish(
                    Float32(data=float(speech_end_latency))
                )

            # Create filenames using the current time.
            timestamp = datetime.now().strftime(
                '%Y%m%d_%H%M%S'
            )

            text_filename = os.path.join(
                self.output_dir,
                f'transcription_{timestamp}.txt',
            )

            # Save the experiment result.
            with open(text_filename, 'w') as text_file:
                text_file.write(
                    f'Inference latency: '
                    f'{inference_latency:.2f}s\n'
                )

                if speech_end_latency is not None:
                    text_file.write(
                        f'Speech-end-to-text latency: '
                        f'{speech_end_latency:.2f}s\n'
                    )

                text_file.write(
                    f'Language: {detected_language}\n'
                )

                text_file.write(
                    f'Transcription: {transcription}\n'
                )

            self.get_logger().info(
                f'Saved transcription to: {text_filename}'
            )

            # Optionally save the raw audio bytes.
            if self.save_audio:
                audio_filename = os.path.join(
                    self.output_dir,
                    f'audio_{timestamp}.raw',
                )

                with open(audio_filename, 'wb') as audio_file:
                    audio_file.write(raw_audio)

        except Exception as error:
            self.get_logger().error(
                f'Error transcribing audio: {error}'
            )

            self.get_logger().error(
                traceback.format_exc()
            )


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = WhisperTranscriptionNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
