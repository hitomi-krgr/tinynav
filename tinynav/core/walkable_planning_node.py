"""Conservative traversability on top of main's planning_node, WITHOUT editing it.
Monkeypatches the module-level build_obstacle_map (sync_callback resolves it at
call time).

Approach A (conservative, unknown = NOT walkable), per-frame (L3 off by default):
  walkable = floor REACHABLE from under the robot (connectivity flood, small
  steps), minus z-span walls, minus small elevated blobs (dog/box). Interior
  holes (enclosed unknown) are filled; affirmative obstacles (z-span | small
  elevated) are re-carved. The exterior frontier (unknown not enclosed) stays
  obstacle -> robot only plans where it has confirmed floor.
  obstacle = ~walkable_final.

WALKABLE_CONF=1 adds the temporal confidence (L3); default OFF (per-frame).

Run:  python3 tinynav/core/walkable_planning_node.py
Env:  WALKABLE_CONF, WALKABLE_DECAY, WALKABLE_GRID_M, WALKABLE_DIAG_LOG.
"""
import os
import time

import numpy as np
import rclpy
from scipy.ndimage import binary_fill_holes

import tinynav.core.planning_node as pn
from tinynav.core.planning_node import PlanningNode, build_obstacle_map
from tinynav.core.traversability import (
    WalkableConfig, WalkableConfidence, extract_height_map, flood_walkable,
    extent_filter, estimate_floor_height, small_elevated_obstacles,
)


class WalkableBuilder:
    """Drop-in for build_obstacle_map -> (NX,NY) bool. Approach A + hole-fill,
    L3 optional."""

    def __init__(self, cfg, decay, use_conf, diag_path):
        self.cfg = cfg
        self.decay = decay
        self.use_conf = use_conf
        self.conf = None
        self._diag_fh = None
        self._last_origin = None
        self._last_ob_n = 0
        self._frame = 0
        if diag_path:
            try:
                self._diag_fh = open(diag_path, "w", buffering=1)
                self._diag_fh.write("frame wall_t origin_x origin_y shift "
                                    "walk_now filled obstacle d_obstacle\n")
            except Exception:
                self._diag_fh = None

    def __call__(self, occupancy_grid, origin, resolution, robot_z, config=None):
        cfg = self.cfg
        nx, ny = occupancy_grid.shape[:2]
        if self.use_conf and self.conf is None:
            self.conf = WalkableConfidence((nx, ny), decay=self.decay)

        zspan = build_obstacle_map(occupancy_grid, origin, resolution, robot_z, config)
        height, observed = extract_height_map(occupancy_grid, origin, resolution, robot_z, cfg)

        ci, cj = nx // 2, ny // 2
        seeds = [(i, j) for i in range(ci - 2, ci + 3) for j in range(cj - 2, cj + 3)
                 if 0 <= i < nx and 0 <= j < ny]
        # floor fallback = robot_z + robot_z_bottom (reuse ObstacleConfig value)
        fallback_drop = -getattr(config, "robot_z_bottom", -0.4) if config else 0.4
        floor_h = estimate_floor_height(height, (ci, cj), radius_cells=10,
                                        robot_z=robot_z, fallback_drop=fallback_drop)

        walk = flood_walkable(height, seeds, floor_h, cfg)
        walk, removed = extent_filter(walk, height, floor_h, resolution, cfg)
        walkable_now = walk & ~zspan

        if self.use_conf:
            not_walkable = (zspan | removed) & observed
            stable = self.conf.update(walkable_now, not_walkable, origin[:2], resolution)
        else:
            stable = walkable_now

        affirmative_obs = zspan | small_elevated_obstacles(height, floor_h, resolution, cfg)
        solid = binary_fill_holes(stable)                 # enclosed unknown -> walkable
        walkable_final = solid & ~affirmative_obs
        for (i, j) in seeds:
            walkable_final[i, j] = True                   # never trap on own footprint

        obstacle = ~walkable_final
        self._diag(origin, walkable_now, solid & ~stable, obstacle)
        return obstacle

    def _diag(self, origin, walkable_now, filled, obstacle):
        if self._diag_fh is None:
            return
        ox, oy = float(origin[0]), float(origin[1])
        shift = 0.0 if self._last_origin is None else float(
            np.hypot(ox - self._last_origin[0], oy - self._last_origin[1]))
        self._last_origin = (ox, oy)
        ob_n = int(obstacle.sum())
        d_ob = ob_n - self._last_ob_n
        self._last_ob_n = ob_n
        self._frame += 1
        self._diag_fh.write(
            f"{self._frame} {time.time():.2f} {ox:.2f} {oy:.2f} {shift:.3f} "
            f"{int(walkable_now.sum())} {int(filled.sum())} {ob_n} {d_ob}\n")


def main(args=None):
    cfg = WalkableConfig()
    decay = float(os.environ.get("WALKABLE_DECAY", "0.99"))
    use_conf = os.environ.get("WALKABLE_CONF", "0") != "0"   # default OFF (per-frame A)
    diag_path = os.environ.get("WALKABLE_DIAG_LOG", "/repo/tool/walkable_test/walkable_diag.log")

    # sync_callback resolves build_obstacle_map from module globals at call time,
    # so replacing the attribute reroutes it. planning_node.py untouched.
    pn.build_obstacle_map = WalkableBuilder(cfg, decay, use_conf, diag_path)

    rclpy.init(args=args)
    node = PlanningNode()

    grid_m = float(os.environ.get("WALKABLE_GRID_M", "10.0"))
    n = int(round(grid_m / node.resolution))
    if n != node.grid_shape[0]:
        node.grid_shape = (n, n, node.grid_shape[2])
        node.origin = np.array(node.grid_shape) * node.resolution / -2.0
        node.occupancy_grid = np.zeros(node.grid_shape)

    node.get_logger().info(
        f"walkable(A, conservative): unknown=obstacle, interior holes filled. "
        f"L3={use_conf} grid={grid_m}m diag={diag_path}")
    try:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
