# Lightweight container to replay a looper bag and visualize the walkable region.
# Runs: (looper_bridge) + walkable_planning_node + walkable_viz_node + foxglove_bridge.
# Avoids the C++ extension (tinynav_cpp_bind) and TensorRT (only map_node/perception
# need those, not the planning/viz pipeline).
FROM ros:humble

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        ros-humble-cv-bridge \
        ros-humble-message-filters \
        ros-humble-foxglove-bridge \
        ros-humble-rosbag2 \
        ros-humble-rosbag2-storage-default-plugins \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# numpy pinned to 1.26.1 for cv_bridge compatibility (matches pyproject.toml).
RUN pip3 install --no-cache-dir \
        "numpy==1.26.1" \
        scipy \
        "numba==0.61.2" \
        opencv-python-headless \
        codetiming \
        fufpy \
        async-lru \
        einops

ENV PYTHONPATH=/repo
WORKDIR /repo
