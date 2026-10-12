cmake_minimum_required(VERSION 3.8)
project(robot_vision_camera)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find required ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "OpenCV version: ${OpenCV_VERSION}")
endif()

# Find GStreamer packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.4)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0>=1.4)

# Check if GStreamer was found
if(GST_FOUND)
  message(STATUS "GStreamer found: ${GST_VERSION}")
else()
  message(FATAL_ERROR "GStreamer not found")
endif()

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# Add compile definitions for GStreamer
add_definitions(${GST_CFLAGS} ${GST_APP_CFLAGS})

# Create executable
add_executable(robot_vision_camera_node
  src/robot_vision_camera.cpp
)

# Target dependencies
ament_target_dependencies(robot_vision_camera_node
  rclcpp
  sensor_msgs
  std_msgs
)

# Link libraries
target_link_libraries(robot_vision_camera_node
  ${OpenCV_LIBS}
  ${GST_LIBRARIES}
  ${GST_APP_LIBRARIES}
)

# Add include directories to target
target_include_directories(robot_vision_camera_node PRIVATE
  ${GST_INCLUDE_DIRS}
  ${GST_APP_INCLUDE_DIRS}
)

# Install targets
install(TARGETS
  robot_vision_camera_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include/
  FILES_MATCHING PATTERN "*.h"
)

# Install launch files (if they exist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch/
  )
endif()

# Install config files (if they exist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config/
  )
endif()

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()