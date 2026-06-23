import numpy as np
import rclpy
from tinynav.core.planning_common import (
    PlanningNodeBase, turn_start_clearance,
    RobotConfig, GO2_CONFIG, B2_CONFIG, ObstacleConfig,
    run_raycasting_loopy, build_obstacle_map,
    generate_trajectory_library_3d, generate_predefined_trajectory_vocabularies,
    score_trajectories_by_ESDF, roll_occupancy_grid,
)

# --- turn-in-the-open tuning ----------------------------------------------
# "Don't start turning inside a narrow corridor; carry forward until the turn
#  point sits in open space." Reward trajectories whose turn START (where yaw
#  first deviates) lands at a high omni-ESDF cell.
TURN_YAW_EPS = np.deg2rad(15.0)   # |yaw - yaw0| above this => "omega took effect"
K_TURN_CLEAR = 0.0


# === PlanningNode class ===
class PlanningNode(PlanningNodeBase):
    """Default planner: turn-in-open term wired in but disabled (K_TURN_CLEAR=0)."""

    def _extra_cost(self, traj, ESDF_map):
        if K_TURN_CLEAR == 0.0:
            return 0.0
        return -K_TURN_CLEAR * turn_start_clearance(
            traj, ESDF_map, self.origin, self.resolution, TURN_YAW_EPS
        )


def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()

    try:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
