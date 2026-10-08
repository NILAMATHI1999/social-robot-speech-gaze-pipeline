cmake_minimum_required(VERSION 3.8)
project(whisper_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(audio_common_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Path to whisper.cpp - adjust this to your whisper.cpp location
set(WHISPER_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../whisper.cpp" CACHE PATH "Path to whisper.cpp directory")

# Find whisper library
find_library(WHISPER_LIBRARY
  NAMES whisper libwhisper
  PATHS ${WHISPER_CPP_DIR}
  PATH_SUFFIXES build build/src src
  REQUIRED
)

# Find ggml library
find_library(GGML_LIBRARY
  NAMES ggml libggml
  PATHS ${WHISPER_CPP_DIR}
  PATH_SUFFIXES build build/ggml ggml build/ggml/src ggml/src
  REQUIRED
)

# Find CUDA ggml library 
find_library(GGML_CUDA_LIBRARY
  NAMES ggml-cuda libggml-cuda
  PATHS ${WHISPER_CPP_DIR}
  PATH_SUFFIXES build/ggml/src/ggml-cuda ggml/src/ggml-cuda
  REQUIRED
)

#message(STATUS "Found whisper library: ${WHISPER_LIBRARY}")
#message(STATUS "Found ggml library: ${GGML_LIBRARY}")
#sommessage(STATUS "Found ggml-cuda library: ${GGML_CUDA_LIBRARY}")

# Include directories
include_directories(
  ${WHISPER_CPP_DIR}/include
  ${WHISPER_CPP_DIR}/src
  ${WHISPER_CPP_DIR}
  ${WHISPER_CPP_DIR}/ggml/include
  ${WHISPER_CPP_DIR}/build/ggml/include
)

# Create executable
add_executable(whisper_transcription_node src/WhisperTranscriptionNode.cpp)

ament_target_dependencies(whisper_transcription_node
  rclcpp
  audio_common_msgs
  std_msgs
)

# Link all libraries including CUDA
target_link_libraries(whisper_transcription_node
  ${WHISPER_LIBRARY}
  ${GGML_LIBRARY}
  ${GGML_CUDA_LIBRARY}
)

# Install
install(TARGETS
  whisper_transcription_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

