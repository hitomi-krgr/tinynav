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

# --- local-target lookahead tuning -----------------------------------------
HEADING_CLIP_RAD = 1.05        # ~60°: stop the lookahead once the global path has bent
                               # this far from its entry direction, so the target stays at
                               # a corner instead of past it (avoids corner-cutting).


# === PlanningNode class ===
class PlanningNode(PlanningNodeBase):
    """Local-target planner: walks the global path and emits a goal-attractor point
    on it, shortening the lookahead at obstacles and corners (an attention clip).
    Lateral obstacle avoidance is left to the base trajectory scorer."""

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
        """Local target = a point ON the global path with the lookahead shortened.

        Not a planner -- an attention/lookahead clip. Lateral obstacle avoidance is
        left to the base trajectory scorer; this only decides *how far ahead* on the
        global path the goal-attractor should sit, so it never lands past an obstacle
        or around a corner (which would make the device cut the corner or swing its
        head toward a target hidden behind a wall). Returns np.ndarray or None.

        Walk the path forward from the robot; stop at the first of:
          - clearance: an on-path cell whose ESDF drops below the safety radius
            (an obstacle has landed on the path) -> last point before it;
          - curvature: the path has bent HEADING_CLIP_RAD from its entry direction
            (a corner) -> the corner itself, so the robot drives to it head-on;
          - the global target.
        """
        if (self.target_pose is None or self.global_path_odom is None
                or len(self.global_path_odom) < 2):
            return None

        init_p_w = self.camera_to_robot_center(T)
        path_xy = self.global_path_odom[:, :2]
        start_idx = int(np.argmin(np.linalg.norm(path_xy - init_p_w[:2], axis=1)))
        target_idx = int(np.argmin(np.linalg.norm(path_xy - self.target_pose[:2], axis=1)))

        step = self.resolution
        safety = self.robot.safety_radius
        target_z = float(self.target_pose[2])
        rows, cols = ESDF_map.shape

        last = None
        entry_dir = None
        for i in range(start_idx, min(target_idx, len(path_xy) - 1)):
            seg = path_xy[i + 1] - path_xy[i]
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            seg_dir = seg / seg_len
            if entry_dir is None:
                entry_dir = seg_dir
            else:
                # Curvature clip: signed angle of this segment vs the entry heading.
                turn = abs(np.arctan2(seg_dir[0] * entry_dir[1] - seg_dir[1] * entry_dir[0],
                                      float(seg_dir @ entry_dir)))
                if turn > HEADING_CLIP_RAD:
                    return last
            n_seg = max(1, int(seg_len / step))
            for s in range(1, n_seg + 1):
                anchor = path_xy[i] + seg * (s / n_seg)
                ex = int((anchor[0] - self.origin[0]) / self.resolution)
                ey = int((anchor[1] - self.origin[1]) / self.resolution)
                sdf_here = ESDF_map[ex, ey] if (0 <= ex < rows and 0 <= ey < cols) else -1.0
                if sdf_here < safety:        # obstacle on the path ahead -> clip before it
                    return last
                last = np.array([anchor[0], anchor[1], target_z])
        return last

    def _resolve_target(self, T, ESDF_map, depth_msg):
        # ESDF-snapped lookahead along global path; falls back to raw target_pose.
        ct = self._compute_local_target(T, ESDF_map)
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
