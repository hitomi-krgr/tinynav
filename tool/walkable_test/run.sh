#!/usr/bin/env bash
# Build the test image and run the walkable-viz pipeline on a bag.
# Usage:  tool/walkable_test/run.sh [BAG_DIR] [RATE]
# View in Foxglove Studio: ws://localhost:8765
#
# Examples:
#   TINYNAV_TRAVERSABILITY=walkable tool/walkable_test/run.sh
#   # debug bag that already has /slam/* (skip bridge, remap infra1->infra2 caminfo):
#   USE_BRIDGE=0 \
#   PLAY_REMAP="--remap /camera/camera/infra1/camera_info:=/camera/camera/infra2/camera_info" \
#   TINYNAV_TRAVERSABILITY=walkable tool/walkable_test/run.sh /path/to/bag
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BAG_DIR="${1:-/Users/hitomikirigiri/Downloads/bag_2026_05_15_17_42_59}"
RATE="${2:-1.0}"
IMG=tinynav-walkable-test

echo "repo: $REPO"
echo "bag : $BAG_DIR"

docker build -t "$IMG" "$REPO/tool/walkable_test"

docker run --rm -it \
    -v "$REPO":/repo \
    -v "$BAG_DIR":/bag \
    -p 8765:8765 \
    -e BAG=/bag \
    -e RATE="$RATE" \
    -e TINYNAV_TRAVERSABILITY="${TINYNAV_TRAVERSABILITY:-}" \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}" \
    -e ROS_LOCALHOST_ONLY=1 \
    -e USE_BRIDGE="${USE_BRIDGE:-1}" \
    -e PLAY_REMAP="${PLAY_REMAP:-}" \
    -e WALKABLE_GRID_M="${WALKABLE_GRID_M:-10.0}" \
    -e WALKABLE_DECAY="${WALKABLE_DECAY:-0.99}" \
    -e WALKABLE_CONF="${WALKABLE_CONF:-0}" \
    "$IMG" \
    bash /repo/tool/walkable_test/pipeline.sh
