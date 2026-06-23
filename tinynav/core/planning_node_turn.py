import numpy as np
import rclpy
from tinynav.core.math_utils import quat_to_matrix
from tinynav.core.planning_common import (
    PlanningNodeBase,
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
K_TURN_CLEAR = 40.0


def turn_start_clearance(traj, ESDF_map, origin, resolution, yaw_eps):
    """ESDF at the point where the trajectory first turns (|yaw - yaw0| > yaw_eps);
    endpoint clearance if it goes straight. ESDF is the omni-directional clearance,
    so this rewards starting the turn in open space rather than inside a narrow
    corridor."""
    rows, cols = ESDF_map.shape

    def yaw_at(row):
        # body +Z is forward -> world heading from the pose rotation
        R = quat_to_matrix(np.array([row[3], row[4], row[5], row[6]]))
        fwd = R @ np.array([0.0, 0.0, 1.0])
        return np.arctan2(fwd[1], fwd[0])

    def esdf_at(xy):
        ex = int((xy[0] - origin[0]) / resolution)
        ey = int((xy[1] - origin[1]) / resolution)
        if 0 <= ex < rows and 0 <= ey < cols:
            return float(ESDF_map[ex, ey])
        return 0.0

    yaw0 = yaw_at(traj[0])
    for i in range(len(traj)):
        dyaw = np.arctan2(np.sin(yaw_at(traj[i]) - yaw0), np.cos(yaw_at(traj[i]) - yaw0))
        if abs(dyaw) > yaw_eps:
            return esdf_at(traj[i, :2])
    return esdf_at(traj[-1, :2])


# === PlanningNode class ===
class PlanningNode(PlanningNodeBase):
    """Turn-in-the-open planner: rewards starting turns at high-clearance cells."""

    def _extra_cost(self, traj, ESDF_map):
        # turn-in-open: reward beginning the turn at a high-clearance cell so
        # the planner carries forward through tight corridors before rotating.
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
