"""Publishes /planning/on_stairs (Bool): is the robot heading into a stair
flight, per the offline capture-path climb labels (path_climb.npy)?

Subscribes /mapping/current_pose_in_map (robot pose in MAP frame, same frame as
poses.npy) and reads the climb label of the nearest capture-path sample, valid
only within PathClimbIndex.assoc_m (~1.5 m, the trajectory-association radius).
Off-path / no map data -> False (=> strict z-span, the safe default).

Consumers: the app backend (frontend indicator) and, later, planning_node
(relax z-span when on stairs, tighten otherwise).
"""
from __future__ import annotations

import argparse
import logging
import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool

from tinynav.core.math_utils import msg2np
from tinynav.core.stair_hint import PathClimbIndex


class StairHintNode(Node):
    def __init__(self, tinynav_map_path: str):
        super().__init__("stair_hint_node")
        self.index: PathClimbIndex | None = None
        path = os.path.join(tinynav_map_path, "path_climb.npy")
        if os.path.exists(path):
            try:
                self.index = PathClimbIndex.load(path)
                n = self.index.pts.shape[0]
                n_climb = int((self.index.pts[:, 3] >= 0.5).sum()) if n else 0
                self.get_logger().info(f"Loaded path_climb.npy: {n} samples, {n_climb} climbing")
            except Exception as e:
                self.get_logger().error(f"Failed to load {path}: {e}")
        else:
            self.get_logger().warn(f"{path} not found; /planning/on_stairs will stay False")

        self.on_stairs_pub = self.create_publisher(Bool, "/planning/on_stairs", 10)
        self.create_subscription(Odometry, "/mapping/current_pose_in_map", self._on_pose, 10)
        self._last = None

    def _on_pose(self, msg: Odometry):
        T, _ = msg2np(msg)
        on = bool(self.index.on_stairs(T[:3, 3])) if self.index else False
        self.on_stairs_pub.publish(Bool(data=on))
        if on != self._last:
            self.get_logger().info(f"on_stairs -> {on}")
            self._last = on


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinynav_map_path", default=os.environ.get("TINYNAV_DB_PATH", "tinynav_db"))
    parsed, _ = parser.parse_known_args()
    rclpy.init(args=args)
    node = StairHintNode(parsed.tinynav_map_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
