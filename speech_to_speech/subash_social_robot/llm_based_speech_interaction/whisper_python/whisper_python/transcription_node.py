#!/usr/bin/env python3

"""
ROS2 Node for real-time audio transcription using faster-whisper.
Subscribes to audio data and transcribes directly without API keys.
"""

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioData
import numpy as np
from faster_whisper import WhisperModel
from datetime import datetime
import os

class WhisperTranscriptionNode(Node):
    def __init__(self):
        super().__init__('whisper_transcription_node')
        
        # Declare parameters
        self.declare_parameter('model_size', 'base')  # tiny, base, small, medium, large-v2, large-v3
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('output_dir', '../recordings')
        self.declare_parameter('save_audio', True)
        self.declare_parameter('device', 'cuda')  # cpu or cuda
        self.declare_parameter('compute_type', 'int8')  # int8, float16, float32
        self.declare_parameter('language', 'en')  # Set to 'None' for auto-detect
        
        # Get parameters
        model_size = self.get_parameter('model_size').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.output_dir = self.get_parameter('output_dir').value
        self.save_audio = self.get_parameter('save_audio').value
        device = self.get_parameter('device').value
        compute_type = self.get_parameter('compute_type').value
        language = self.get_parameter('language').value
        self.language = language if language != 'None' else None
        
        # Load Whisper model
        self.get_logger().info(f'Loading faster-whisper model ({model_size}) on {device}...')
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type
        )
        self.get_logger().info('Model loaded successfully!')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subscribe to audio data
        self.subscription = self.create_subscription(
            AudioData,
            'speech_audio',
            self.audio_callback,
            10
        )
        
        self.get_logger().info('Whisper Transcription Node initialized')
        #self.get_logger().info(f'Listening on topic: speech_audio')
        #self.get_logger().info(f'Output directory: {self.output_dir}')
    
    def audio_callback(self, msg):
        self.get_logger().info('Received audio data, transcribing...')
        
        start_time = datetime.now()
        
        try:
            # Convert raw audio data to numpy array
            audio_data = np.frombuffer(bytes(msg.data), dtype=np.int16)
            
            # Normalize audio to float32 in range [-1, 1]
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            #self.get_logger().info('Transcribing with faster-whisper...')
            
            segments, info = self.model.transcribe(
                audio_float,
                language=self.language,
                beam_size=5
            )
            
            # Collect all segments into full transcription
            transcription = " ".join([segment.text for segment in segments]).strip()
            detected_language = info.language
            self.get_logger().info(f'Transcription: {transcription}')
            
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            self.get_logger().info(f'Transcription completed in {latency:.2f} seconds')
            #self.get_logger().info(f'Detected language: {detected_language}')
            
            # Save transcription
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            txt_filename = os.path.join(self.output_dir, f'transcription_{timestamp}.txt')
            
            with open(txt_filename, 'w') as f:
                f.write(f"Latency: {latency:.2f}s\n")
                f.write(f"Language: {detected_language}\n")
                f.write(f"Transcription: {transcription}\n")
            
            self.get_logger().info(f'Saved transcription to: {txt_filename}')
            
            # Optionally save the audio file AFTER transcription
            if self.save_audio:
                wav_filename = os.path.join(self.output_dir, f'audio_{timestamp}.raw')
                with open(wav_filename, 'wb') as f:
                    f.write(audio_data.tobytes())
                #self.get_logger().info(f'Saved audio to: {wav_filename}')
            
        except Exception as e:
            self.get_logger().error(f'Error transcribing audio: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WhisperTranscriptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()