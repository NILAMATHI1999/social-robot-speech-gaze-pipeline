from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.expanduser('/home/robot/ament_ws/src/whisper.cpp/models/ggml-large-v3-turbo.bin'), #base.en.bin
        description='Path to Whisper model file'
    )
    
    n_threads_arg = DeclareLaunchArgument(
        'n_threads',
        default_value='4',
        description='Number of threads for Whisper processing'
    )
    
    language_arg = DeclareLaunchArgument(
        'language',
        default_value='en',
        description='Language code for transcription'
    )
    
    translate_arg = DeclareLaunchArgument(
        'translate',
        default_value='false',
        description='Whether to translate to English'
    )
    
    save_files_arg = DeclareLaunchArgument(
        'save_files',
        default_value='true',
        description='Whether to save audio and transcription files'
    )
    
    save_directory_arg = DeclareLaunchArgument(
        'save_directory',
        default_value=os.path.expanduser('/home/robot/recordings/'),
        description='Directory to save recordings'
    )
    
    sample_rate_arg = DeclareLaunchArgument(
        'sample_rate',
        default_value='16000',
        description='Audio sample rate in Hz'
    )
    
    # Create node
    whisper_node = Node(
        package='whisper_ros2',
        executable='whisper_transcription_node',
        name='whisper_transcription_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'n_threads': LaunchConfiguration('n_threads'),
            'language': LaunchConfiguration('language'),
            'translate': LaunchConfiguration('translate'),
            'save_files': LaunchConfiguration('save_files'),
            'save_directory': LaunchConfiguration('save_directory'),
            'sample_rate': LaunchConfiguration('sample_rate'),
        }]
    )
    
    return LaunchDescription([
        model_path_arg,
        n_threads_arg,
        language_arg,
        translate_arg,
        save_files_arg,
        save_directory_arg,
        sample_rate_arg,
        whisper_node
    ])
