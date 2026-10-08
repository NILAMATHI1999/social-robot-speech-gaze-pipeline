## 🧱 Project Structure

This project works in collaboration with two other packages that are not included here.

1) The ROS2 audio_common package

This package can be clone from the repository with the command "git clone https://github.com/hello-chintan/audio_common.git" 

2) The Whisper.cpp package

This package needs to be cloned and started independently. The documentation is found here: https://github.com/ggml-org/whisper.cpp/tree/master

## 🧩 Installation 

```
mkdir -p ~/ament_ws/src 
cd ~/ament_ws/src
git clone <gitlab-url>  # clone this repository
git clone https://github.com/hello-chintan/audio_common.git
git clone https://github.com/ggml-org/whisper.cpp.git
```

For the **_respeaker_ros2_** package, 

- Register respeaker udev rules to access USB device without permission from user space.

```
cd ~/ament_ws/src/llm_based_speech_interaction/respeaker_ros2
sudo cp -f ~/ament_ws/src/llm_based_speech_interaction/respeaker_ros2/config/60-respeaker.rules /etc/udev/rules.d/60-respeaker.rules
sudo systemctl restart udev
```
- Install Python requirements

```
cd ~/ament_ws/src/respeaker_ros2
sudo pip install -r requirements.txt
```
- Update firmware

```
git clone https://github.com/respeaker/usb_4_mic_array.git
cd usb_4_mic_array
sudo python dfu.py --download 6_channels_firmware.bin  # The 6 channels version 
```

More information on the respeaker_ros package can be obtained here: https://github.com/hello-chintan/respeaker_ros2?tab=readme-ov-file

For the **_Whisper_cpp_** package,

- Navigate into the directory

`cd whisper.cpp`

- Then, download one of the Whisper models converted in ggml format. For example:

`sh ./models/download-ggml-model.sh base.en`

- Now build the project

```
cmake -B build
cmake --build build -j --config Release
```

More information can be obtained from this repository: https://github.com/ggml-org/whisper.cpp/tree/master

## 🧩 Build

Build the project by running the following commands

```
cd ament_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -i -r -n -y
colcon build --packages-select audio_common_msgs audio_capture audio_play sound_play speech_recogntion_msgs
source install/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 🚀 Execution

There are three whisper packages which can be used to carryout the transcription of the recording or audio input gotten from the respeaker: 

1. whisper_ros2 package: This package makes use of the whisper.cpp (https://github.com/ggml-org/whisper.cpp). A plain C/C++ implementaion of OpenAI whisper without dependencies. It uses in-memory transcription, that is, it calls the openai API on the audio recorder before downloading the audio in a wav.

2. respeaker_ros2_cpp package: It uses the whisper.cpp package as above, but here the audio is first downloaded in a wav file and then the wav audio file is then transcripted.

3. whisper_python package: This package uses faster-whisper (https://github.com/SYSTRAN/faster-whisper). It is a reimplementation of OpenAI's Whisper model using CTranslate2

A) C++ Transcription


To successfully launch the project using the **whisper_ros2** package, change the model_path argument in the file `whisper_ros2/launch/whisper.transcription.py`

Or include the model_path as an argument to the run command

#### ⚙️ Launch Parameters

| Argument         | Description                                        | Default Value  |
|------------------|----------------------------------------------------|----------------|
| `model_path` | The path to the downloaded whisper model file                                | base.en     |
| `language_en`   | The language code for transcription   | en       |

**In one terminal, run the respeaker_ros2 package**
```
source install/setup.bash
ros2 launch respeaker_ros2 respeaker.launch.py
```

**Now run the whisper_cpp package (package that does in-memory transcription) in another terminal**
```
source install/setup.bash

# Disable file saving
ros2 launch whisper_ros2 whisper_transcription.launch.py save_files:=false

# Custom save directory
ros2 launch whisper_ros2 whisper_transcription.launch.py save_directory:=/my/custom/path

# Launch with model_path as argument
ros2 launch whisper_ros2 whisper_transcription.launch.py model_path:=/my/model/path/ggml-base.en.bin 
```

**You can also run the respeaker_ros_cpp package (package the downloads the audio first, before transcription) in another terminal**
```
source install/setup.bash
ros2 run respeaker_ros2_cpp audio_recorder_node
```
Note: some paths need to be changed here to reflect the path where the downloads should be stored and where the whisper model is found

B) Python Transcription

Here the transcription is carried out with the faster-whisper api, using the **whisper_python** package

**In one terminal, run the respeaker_ros2 package**
```
source install/setup.bash
ros2 launch respeaker_ros2 respeaker.launch.py
```

**In another terminal, run the whisper_python package**

```
source install/setup.bash

# Launch with model_size as argument if necessary

ros2 run whisper_python transcription_node --ros-args -p model_size:="base"

```
