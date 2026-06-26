import numpy as np
import rclpy
import tf2_ros
from rclpy.time import Time
from rclpy.duration import Duration
from nav_msgs.msg import Path, Odometry
from tinynav.core.math_utils import quat_to_matrix
from tinynav.core.planning_common import (
    PlanningNodeBase,
    RobotConfig, GO2_CONFIG, B2_CONFIG, ObstacleConfig,
    run_raycasting_loopy, build_obstacle_map,
    generate_trajectory_library_3d, generate_predefined_trajectory_vocabularies,
    score_trajectories_by_ESDF, roll_occupancy_grid,
)

# Lookahead point selection on the global path.
SMOOTH_M       = 0.5            # arc length used to smooth the path heading (kills A* grid jaggedness)
LOOKAHEAD_MAX  = 2.0            # extend the aim this far on straight/gentle path (~traj horizon -> full speed)
CORNER_ANGLE   = np.deg2rad(60) # only a bend sharper than this counts as a corner to stop the aim at;
                                # gentler bends are treated as straight so the aim stays long (fast, smooth).

# === PlanningNode class ===
class PlanningNode(PlanningNodeBase):
    """Local-target planner: aims at the furthest point on the global path that the
    robot can head straight to without cutting a corner. This keeps the route on the
    polyline instead of slicing across bends; obstacle avoidance stays with the base
    trajectory scorer."""

    def _setup_extras(self):
        self._last_local_target = None
        self.global_path_odom = None  # /mapping/global_plan transformed into world frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.local_target_pub = self.create_publisher(Odometry, '/planning/local_target', 10)
        self.create_subscription(Path, '/mapping/global_plan', self._on_global_path_map, 10)

    def poi_change_callback(self, msg):
        self.target_pose = None
        self._last_local_target = None
        self.global_path_odom = None

    def _on_global_path_map(self, msg):
        pts = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses])
        if len(pts) == 0:
            self.global_path_odom = None
            return
        try:
            tf = self.tf_buffer.lookup_transform("world", "map", Time(), timeout=Duration(seconds=0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        t, q = tf.transform.translation, tf.transform.rotation
        T_om = np.eye(4)
        T_om[:3, :3] = quat_to_matrix(np.array([q.x, q.y, q.z, q.w]))
        T_om[:3, 3] = [t.x, t.y, t.z]
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        self.global_path_odom = (T_om @ pts_h.T).T[:, :3]

    def _compute_local_target(self, T):
        """Local target = a lookahead point on the global path.

        The base DWA only pulls a trajectory's *endpoint* toward one goal point, so
        the aim point governs both heading and speed: too far past a corner makes the
        robot slice across it (Bezier); too near makes the matched arc slow. So extend
        the aim along the path up to LOOKAHEAD_MAX (~the trajectory horizon, for full
        speed) on straight/gently-curving path, and stop earlier only at a real corner
        (heading bent > CORNER_ANGLE) so the robot drives to the corner instead of
        cutting it. Obstacle avoidance is NOT done here -- that is the scorer's job.
        Returns np.ndarray or None.
        """
        if (self.target_pose is None or self.global_path_odom is None
                or len(self.global_path_odom) < 2):
            return None

        robot = self.camera_to_robot_center(T)[:2]
        path_xy = self.global_path_odom[:, :2]
        n = len(path_xy)
        start_idx = int(np.argmin(np.linalg.norm(path_xy - robot, axis=1)))
        target_idx = int(np.argmin(np.linalg.norm(path_xy - self.target_pose[:2], axis=1)))
        target_z = float(self.target_pose[2])
        if target_idx <= start_idx:
            return None

        # Arc length from path[0]; used for both the smoothing window and the cap.
        cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))))

        def smoothed_dir(i):
            """Unit path heading at i, averaged over SMOOTH_M ahead (ignores grid jaggedness)."""
            j = i
            while j < n - 1 and cum[j] - cum[i] < SMOOTH_M:
                j += 1
            d = path_xy[j] - path_xy[i]
            L = float(np.linalg.norm(d))
            return d / L if L > 1e-6 else None

        entry_dir = smoothed_dir(start_idx)
        aim_idx = start_idx
        for k in range(start_idx + 1, target_idx + 1):
            if cum[k] - cum[start_idx] >= LOOKAHEAD_MAX:   # far enough for full speed
                aim_idx = k
                break
            dir_k = smoothed_dir(k)
            if entry_dir is not None and dir_k is not None:
                turn = abs(np.arctan2(dir_k[0] * entry_dir[1] - dir_k[1] * entry_dir[0],
                                      float(dir_k @ entry_dir)))
                if turn >= CORNER_ANGLE:                   # real corner -> stop at it, don't cut
                    aim_idx = k
                    break
            aim_idx = k

        aim = path_xy[aim_idx]
        return np.array([aim[0], aim[1], target_z])

    def _resolve_target(self, T, ESDF_map, depth_msg):
        # Corner-respecting lookahead on the global path; falls back to raw target_pose.
        ct = self._compute_local_target(T)
        local_target = ct if ct is not None else self.target_pose

        # Temporal low-pass; only smooth the local-target output, not the raw fallback.
        if ct is not None:
            if self._last_local_target is not None:
                jump = float(np.linalg.norm(local_target[:2] - self._last_local_target[:2]))
                if jump < 1.5:
                    local_target = 0.3 * local_target + 0.7 * self._last_local_target
            self._last_local_target = local_target

        if local_target is not None:
            lt_msg = Odometry()
            lt_msg.header.stamp = depth_msg.header.stamp
            lt_msg.header.frame_id = "world"
            lt_msg.pose.pose.position.x = float(local_target[0])
            lt_msg.pose.pose.position.y = float(local_target[1])
            lt_msg.pose.pose.position.z = float(local_target[2])
            lt_msg.pose.pose.orientation.w = 1.0
            self.local_target_pub.publish(lt_msg)

        return local_target


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
