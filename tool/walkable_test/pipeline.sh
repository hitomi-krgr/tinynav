#!/usr/bin/env bash
# In-container launcher: bring up the walkable-viz pipeline on a bag.
# Consumers first, then bag play, so camera_info / first frames aren't missed.
set -m
source /opt/ros/humble/setup.bash
export PYTHONPATH=/repo:${PYTHONPATH}
# Isolated DDS: unique domain + localhost-only so other LAN devices don't leak in.
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
export ROS_LOCALHOST_ONLY=1

BAG=${BAG:-/bag}
RATE=${RATE:-1.0}
USE_BRIDGE=${USE_BRIDGE:-1}     # 0 = bag already has /slam/* (debug bag) -> skip bridge
PLAY_REMAP=${PLAY_REMAP:-}      # extra `ros2 bag play` args (e.g. --remap a:=b)

PID_BRIDGE=""
if [ "$USE_BRIDGE" != "0" ]; then
    echo "[pipeline] looper_bridge ..."
    python3 /repo/tool/looper_bridge_node.py &
    PID_BRIDGE=$!
    sleep 4
else
    echo "[pipeline] skipping bridge (bag provides /slam/*); play args: $PLAY_REMAP"
fi

if [ "${TINYNAV_TRAVERSABILITY:-}" = "walkable" ]; then
    echo "[pipeline] walkable_planning_node ..."
    python3 /repo/tinynav/core/walkable_planning_node.py &
else
    echo "[pipeline] planning_node (z-span baseline) ..."
    python3 /repo/tinynav/core/planning_node.py &
fi
PID_PLAN=$!
sleep 4

echo "[pipeline] walkable_viz_node ..."
python3 /repo/tinynav/core/walkable_viz_node.py &
PID_VIZ=$!
sleep 1

echo "[pipeline] foxglove_bridge on :8765 ..."
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 &
PID_FOX=$!
sleep 2

cleanup() {
    echo "[pipeline] stopping ..."
    kill $PID_BRIDGE $PID_PLAN $PID_VIZ $PID_FOX 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "[pipeline] playing bag $BAG at rate $RATE (loop) ..."
echo "[pipeline] >>> connect Foxglove Studio to ws://localhost:8765 <<<"
ros2 bag play "$BAG" --rate "$RATE" --loop $PLAY_REMAP
