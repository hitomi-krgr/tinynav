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

# --- centerline (ESDF-snapped local target) tuning (PR145) -----------------
LOOKAHEAD_MAX_M = 4.0          # lookahead when robot heading aligned with bearing
LOOKAHEAD_MIN_M = 1.0          # lookahead when heading orthogonal/opposite to bearing
LOOKAHEAD_DECAY = 4.0          # exp decay of lookahead vs heading-bearing mismatch
LATERAL_RANGE_M = 0.4          # ± lateral search range for max-SDF snap
LATERAL_SAMPLES = 9            # samples across the lateral range
CLEARANCE_BRAKE_M = 0.4        # stop walking the centerline once clearance drops here


# === PlanningNode class ===
class PlanningNode(PlanningNodeBase):
    """Centerline planner: walks the global path, snapping laterally to the
    most-open ESDF cell, to produce a local target that hugs corridor centers."""

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

    def _compute_local_target(self, T, ESDF_map):
        """ESDF-snapped lookahead along the global path. Returns np.ndarray or None.

        - Lateral ±LATERAL_RANGE_M snap to max-SDF cell -> centers in free space.
        - Lookahead shortens as robot heading diverges from target bearing.
        - Walk caps at target_pose; clearance brake terminates near obstacles.
        """
        if (self.target_pose is None or self.global_path_odom is None
                or len(self.global_path_odom) < 2):
            return None

        init_p_w = self.camera_to_robot_center(T)
        path_xy = self.global_path_odom[:, :2]
        start_idx = int(np.argmin(np.linalg.norm(path_xy - init_p_w[:2], axis=1)))
        target_idx = int(np.argmin(np.linalg.norm(path_xy - self.target_pose[:2], axis=1)))

        # Body frame is +Z forward (see _front_obstacle_dist / footprint).
        fwd_w = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        robot_yaw = np.arctan2(fwd_w[1], fwd_w[0])
        bearing = np.arctan2(self.target_pose[1] - init_p_w[1],
                             self.target_pose[0] - init_p_w[0])
        ang = abs(np.arctan2(np.sin(bearing - robot_yaw), np.cos(bearing - robot_yaw)))
        max_dist = float(LOOKAHEAD_MIN_M +
                         (LOOKAHEAD_MAX_M - LOOKAHEAD_MIN_M) * np.exp(-LOOKAHEAD_DECAY * ang))

        step = self.resolution
        safety = self.robot.safety_radius
        target_z = float(self.target_pose[2])
        rows, cols = ESDF_map.shape
        lats = np.linspace(-LATERAL_RANGE_M, LATERAL_RANGE_M, LATERAL_SAMPLES)

        last = None
        accumulated = 0.0
        for i in range(start_idx, min(target_idx, len(path_xy) - 1)):
            seg = path_xy[i + 1] - path_xy[i]
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            perp = np.array([-seg[1] / seg_len, seg[0] / seg_len])
            n_seg = max(1, int(seg_len / step))
            for s in range(1, n_seg + 1):
                anchor = path_xy[i] + seg * (s / n_seg)
                # Vectorize the LATERAL_SAMPLES SDF lookups across this anchor.
                candidates = anchor + perp * lats[:, None]
                ex = ((candidates[:, 0] - self.origin[0]) / self.resolution).astype(int)
                ey = ((candidates[:, 1] - self.origin[1]) / self.resolution).astype(int)
                valid = (ex >= 0) & (ex < rows) & (ey >= 0) & (ey < cols)
                sdfs = np.where(
                    valid,
                    ESDF_map[np.clip(ex, 0, rows - 1), np.clip(ey, 0, cols - 1)],
                    -1.0,
                )
                k = int(np.argmax(sdfs))
                best_sdf = float(sdfs[k])
                if best_sdf >= safety:
                    last = np.array([float(candidates[k, 0]), float(candidates[k, 1]), target_z])
                    if best_sdf < CLEARANCE_BRAKE_M:
                        return last
                accumulated += step
                if accumulated >= max_dist:
                    return last
        return last

    def _resolve_target(self, T, ESDF_map, depth_msg):
        # ESDF-snapped lookahead along global path; falls back to raw target_pose.
        ct = self._compute_local_target(T, ESDF_map)
        local_target = ct if ct is not None else self.target_pose

        # Temporal low-pass; only smooth the centerline output, not the raw fallback.
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
