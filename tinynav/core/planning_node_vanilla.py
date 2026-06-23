import rclpy
from tinynav.core.planning_common import (
    PlanningNodeBase,
    RobotConfig, GO2_CONFIG, B2_CONFIG, ObstacleConfig,
    run_raycasting_loopy, build_obstacle_map,
    generate_trajectory_library_3d, generate_predefined_trajectory_vocabularies,
    score_trajectories_by_ESDF, roll_occupancy_grid,
)


# === PlanningNode class ===
class PlanningNode(PlanningNodeBase):
    """Vanilla planner: cost = clearance + distance-to-goal, no extra terms."""
    pass


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
