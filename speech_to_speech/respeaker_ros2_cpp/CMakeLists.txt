cmake_minimum_required(VERSION 3.5)
project(respeaker_ros2_cpp)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(audio_common_msgs REQUIRED)

# Find libsndfile via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

# Include whisper.cpp
include_directories(${CMAKE_SOURCE_DIR}/../../whisper.cpp)
add_subdirectory(${CMAKE_SOURCE_DIR}/../../whisper.cpp whisper_build EXCLUDE_FROM_ALL)

# Include directories
include_directories(
  include
  ${PORTAUDIO_INCLUDE_DIRS}
  ${SNDFILE_INCLUDE_DIRS}
)

# === Add your executable BEFORE linking ===
add_executable(audio_recorder_node src/audio_recorder_node.cpp)

# Link ROS2 + sndfile libraries + whisper.cpp
ament_target_dependencies(audio_recorder_node rclcpp audio_common_msgs std_msgs)
target_link_libraries(audio_recorder_node 
  whisper
  ${SNDFILE_LIBRARIES})

# Install
install(TARGETS
  audio_recorder_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
