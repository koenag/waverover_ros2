FROM ros:jazzy-ros-core
RUN apt-get update && apt-get install -y \
ros-jazzy-demo-nodes-cpp \
ros-jazzy-foxglove-bridge \
ros-jazzy-tf2-ros
