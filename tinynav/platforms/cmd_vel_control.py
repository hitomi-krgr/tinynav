import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation as R
import numpy as np
import logging
import time

# Module-level logger for cases where self.get_logger() is not available
logger = logging.getLogger(__name__)

class CmdVelControlNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_control_node')
        self.logger = self.get_logger()  # Use ROS2 logger
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Odometry, '/slam/odometry', self.pose_callback, 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)
        self.T_robot_to_camera = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]]
        )
        # The odometry is the CAMERA pose, but planner path poses are CONTROL-CENTER
        # poses and the camera sits this far AHEAD of the control center
        # (B2: camera_x - control_x = 0.5 - (-0.5) = 1.0 m). The closed-loop heading
        # MUST be referenced at the control center, otherwise the short (~1 m)
        # control-center path lies behind the camera -> heading_err ~ +/-pi -> the
        # robot rotates in place forever and cannot move. (Diagnosed from a bag.)
        self.cam_forward_offset = 1.0
        self.T_camera_to_control = self.T_robot_to_camera.copy()
        self.T_camera_to_control[2, 3] = -self.cam_forward_offset  # back along camera +z (=forward)
        self.last_path_time = 0.0
        self.pose = None
        self.path = None

        # === Control loop (ported from planning_node_compare style) ===
        # Planner input is typically 7-10 Hz; over-driving cmd publish rate amplifies jitter.
        self.cmd_rate_hz = 12.0
        # Use minima; actual stale thresholds are scaled by observed planner period.
        self.path_stale_slow_s = 0.35
        self.path_stale_stop_s = 0.8
        self.path_stale_slow_factor = 3.5
        self.path_stale_stop_factor = 5.0
        self.max_linear_acc = 0.6   # m/s^2
        self.max_angular_acc = 0.8  # rad/s^2
        self.max_angular_speed = 0.8  # rad/s
        self.planner_dt = 0.1       # trajectory dt in planning_node
        # planning_node publishes path with for j in range(..., step=10), so points are ~1.0 s apart.
        self.path_pose_stride = 10
        self.path_period_ema = 0.12
        self.path_filter_tau = 0.30
        self.lookahead_steps = 1
        self.lookahead_distance = 0.8
        self.yaw_kp = 0.4
        self.yaw_p_ff_max = 0.4
        self.yaw_kd = 0.0
        self._prev_heading_err = None
        # Static-friction compensation: very small vx often cannot move the robot.
        self.min_effective_linear_speed = 0.1
        self.min_effective_angular_speed = 0.1
        self.linear_engage_threshold = 0.04
        self.fixed_reverse_speed = 0.2
        self.force_turn_heading_threshold = np.deg2rad(80.0)
        self.rotate_first_heading_threshold = 0.45
        self.rotate_first_gain = 1.6

        self.latest_cmd = Twist()
        self.target_point_world = None
        self.path_vyaw_ff = 0.0
        self.is_backward_segment = False
        self.prev_cmd = Twist()
        self.last_cmd_pub_time = time.monotonic()
        self.last_path_update_time = None
        self._paused = False
        self._nav_active = False
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/nav/paused', self._on_paused, _latched_qos)
        self.create_subscription(Bool, '/nav/active', self._on_nav_active, _latched_qos)
        self.cmd_timer = self.create_timer(1.0 / self.cmd_rate_hz, self.cmd_timer_callback)

    def _on_paused(self, msg: Bool):
        self._paused = msg.data
        if not self._paused:
            # Reset prev_cmd so resume starts from zero cleanly
            self.prev_cmd = Twist()

    def _on_nav_active(self, msg: Bool):
        was_active = self._nav_active
        self._nav_active = bool(msg.data)
        if was_active and not self._nav_active:
            self.latest_cmd = Twist()
            self.prev_cmd = Twist()
            self.last_path_update_time = None
            # Send one stop when navigation is deactivated, then stay silent so
            # manual teleop can own /cmd_vel without being overwritten by zeros.
            self.cmd_pub.publish(Twist())

    def pose_callback(self, msg):
        self.pose = msg

    def _clamp_step(self, target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    @staticmethod
    def _pose_to_T(pose_msg) -> np.ndarray:
        T = np.eye(4)
        position = pose_msg.pose.position
        rot = pose_msg.pose.orientation
        quat = [rot.x, rot.y, rot.z, rot.w]
        T[:3, :3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = np.array([position.x, position.y, position.z]).ravel()
        return T

    @staticmethod
    def _pose_xy(pose_msg) -> np.ndarray:
        """Just the (x, y) position — avoids quaternion math when only the
        translation is needed (e.g. the lookahead distance scan)."""
        p = pose_msg.pose.position
        return np.array([p.x, p.y])

    def _actual_yaw(self):
        """World heading of the robot's measured forward axis (odometry)."""
        if self.pose is None:
            return None
        fwd = self._pose_to_T(self.pose.pose)[:3, :3] @ np.array([0.0, 0.0, 1.0])  # optical +z = forward
        return float(np.arctan2(fwd[1], fwd[0]))

    def _path_intended_yaw(self):
        """World heading the PLAN intends here: forward axis of the published
        trajectory pose nearest the control center. None if no path/pose. This is the
        drift reference — comparing it to the measured heading isolates open-loop yaw
        drift from the planner's deliberate turning (which the feedforward handles)."""
        if self.pose is None or self.path is None or len(self.path.poses) == 0:
            return None
        ctrl_xy = (self._pose_to_T(self.pose.pose) @ self.T_camera_to_control)[:2, 3]
        best_i, best_d = 0, float('inf')
        for i, ps in enumerate(self.path.poses):
            d = (ps.pose.position.x - ctrl_xy[0]) ** 2 + (ps.pose.position.y - ctrl_xy[1]) ** 2
            if d < best_d:
                best_d, best_i = d, i
        fwd = self._pose_to_T(self.path.poses[best_i])[:3, :3] @ np.array([0.0, 0.0, 1.0])
        return float(np.arctan2(fwd[1], fwd[0]))

    def cmd_timer_callback(self):
        now = time.monotonic()
        dt = max(1e-3, now - self.last_cmd_pub_time)
        self.last_cmd_pub_time = now

        if not self._nav_active:
            return

        if self._paused:
            self.cmd_pub.publish(Twist())
            self.prev_cmd = Twist()
            return

        # Stale-path protection: slow down, then stop if planner has not refreshed.
        age = float('inf') if self.last_path_update_time is None else (now - self.last_path_update_time)
        stale_slow_s = max(self.path_stale_slow_s, self.path_period_ema * self.path_stale_slow_factor)
        stale_stop_s = max(self.path_stale_stop_s, self.path_period_ema * self.path_stale_stop_factor)
        target_cmd = Twist()
        target_cmd.linear.x = self.latest_cmd.linear.x

        # Closed-loop yaw: recompute heading error against the LIVE pose every tick,
        # P term + path feedforward. Falls back to the path-derived feedforward when
        # no live heading is available (no pose / no target yet).
        # Closed-loop yaw = planner FEEDFORWARD omega + a small P that only cancels
        # open-loop HEADING DRIFT. The reference is the planned trajectory's intended
        # heading (orientation of the nearest published pose), NOT the bearing to a
        # lookahead point. So the planner decides how much to turn (feedforward); the
        # loop only nulls the difference between the heading we actually achieved and
        # the heading the plan intended — i.e. it removes per-device open-loop drift
        # without re-deciding the motion. No rotate-in-place / force-turn gates.
        intended_yaw = self._path_intended_yaw()
        actual_yaw = self._actual_yaw()
        if intended_yaw is not None and actual_yaw is not None:
            drift = float(np.arctan2(np.sin(actual_yaw - intended_yaw),
                                     np.cos(actual_yaw - intended_yaw)))
            ddrift = 0.0 if self._prev_heading_err is None else (drift - self._prev_heading_err) / dt
            self._prev_heading_err = drift
            vyaw = self.path_vyaw_ff - (self.yaw_kp * drift + self.yaw_kd * ddrift)
        else:
            # No path/pose yet: pure feedforward.
            vyaw = self.path_vyaw_ff
            self._prev_heading_err = None
        target_cmd.angular.z = float(np.clip(vyaw, -self.max_angular_speed, self.max_angular_speed))

        if age > stale_stop_s:
            target_cmd.linear.x = 0.0
            target_cmd.angular.z = 0.0
        elif age > stale_slow_s:
            target_cmd.linear.x *= 0.3
            target_cmd.angular.z *= 0.5

        out = Twist()
        out.linear.y = 0.0

        # Reverse is a predefined planner vocabulary: straight back at fixed speed.
        # Do not smooth or re-lock it here; just pass it through while stale/paused guards still work.
        if target_cmd.linear.x < 0.0:
            out.linear.x = target_cmd.linear.x
            out.angular.z = 0.0
            self.cmd_pub.publish(out)
            self.prev_cmd = out
            return

        # Forward/turning commands still get acceleration limiting and robot minimum-speed locks.
        max_dv = self.max_linear_acc * dt
        # If we just left reverse mode, do not let acceleration limiting leak another reverse command.
        prev_linear_x = 0.0 if self.prev_cmd.linear.x < 0.0 else self.prev_cmd.linear.x
        out.linear.x = self._clamp_step(target_cmd.linear.x, prev_linear_x, max_dv)
        # Do not acceleration-limit yaw. The planner/control layer already decides the turn rate,
        # and forced rotate-in-place should take effect immediately.
        out.angular.z = float(np.clip(target_cmd.angular.z, -self.max_angular_speed, self.max_angular_speed))

        # Linear x: robot cannot execute tiny non-zero speeds reliably. When the
        # planner asks for ANY forward motion (target > 0), creep at +min instead of
        # snapping to 0 — otherwise a planner command in (0, min) freezes the robot,
        # which then never advances, never updates its lookahead, and deadlocks (e.g.
        # at a wall-adjacent waypoint). Only a non-positive target decays to 0.
        if 0.0 < out.linear.x < self.min_effective_linear_speed:
            out.linear.x = self.min_effective_linear_speed if target_cmd.linear.x > 0.0 else 0.0
        elif abs(out.linear.x) < self.min_effective_linear_speed:
            out.linear.x = 0.0

        # Angular z: same idea; tiny requested turns snap to executable min, decays snap to 0.
        if 0.0 < abs(out.angular.z) < self.min_effective_angular_speed:
            if abs(target_cmd.angular.z) >= self.min_effective_angular_speed:
                out.angular.z = float(np.sign(target_cmd.angular.z) * self.min_effective_angular_speed)
            else:
                out.angular.z = 0.0

        self.cmd_pub.publish(out)
        self.prev_cmd = out
        
    def path_callback(self, msg):
        if not self._nav_active:
            return
        if msg is None or self.pose is None:
            return
        if len(msg.poses) < 2:
            return
        self.path = msg

        ros_now = self.get_clock().now().to_msg()
        self.last_path_time = ros_now.sec + ros_now.nanosec * 1e-9
        now_mono = time.monotonic()
        if self.last_path_update_time is not None:
            period = np.clip(now_mono - self.last_path_update_time, 0.05, 0.5)
            self.path_period_ema = 0.85 * self.path_period_ema + 0.15 * float(period)
        self.last_path_update_time = now_mono

        T1 = self._pose_to_T(self.path.poses[0])
        # Choose the lookahead pose by distance from the CONTROL CENTER (the heading-
        # error origin), so it is consistent with where the path is measured. Fall back
        # to lookahead_steps as a floor and the last pose as a cap.
        ctrl_xy = (self._pose_to_T(self.pose.pose) @ self.T_camera_to_control)[:2, 3]
        step_idx = int(min(self.lookahead_steps, len(self.path.poses) - 1))
        for j in range(1, len(self.path.poses)):
            if np.linalg.norm(self._pose_xy(self.path.poses[j]) - ctrl_xy) >= self.lookahead_distance:
                step_idx = max(step_idx, j)
                break
        else:
            step_idx = len(self.path.poses) - 1
        T2 = self._pose_to_T(self.path.poses[step_idx])
        T_robot_1 = T1 @ self.T_robot_to_camera
        T_robot_2 = T2 @ self.T_robot_to_camera
        T_robot_2_to_1 = np.linalg.inv(T_robot_1) @ T_robot_2
        p = T_robot_2_to_1[:3, 3]
        # dt must match actual spacing between published Path poses, not raw trajectory dt.
        dt = self.planner_dt * self.path_pose_stride * max(1, step_idx)
        linear_velocity_vec = p / dt
        r = R.from_matrix(T_robot_2_to_1[:3, :3])
        angular_velocity_vec = r.as_rotvec() / dt

        raw_vx = float(linear_velocity_vec[0])
        is_backward_segment = raw_vx < 0.0
        if is_backward_segment:
            vx = -self.fixed_reverse_speed
        else:
            vx = float(np.clip(raw_vx, 0.0, 0.5))

        # Store the world-frame lookahead target + path feedforward yaw. The closed-loop
        # heading P term is applied in cmd_timer_callback against the live odometry pose,
        # so per-device open-loop yaw drift is corrected continuously.
        self.target_point_world = (T_robot_2[:3, 3]).copy()
        self.is_backward_segment = is_backward_segment
        self.path_vyaw_ff = 0.0 if is_backward_segment else float(angular_velocity_vec[2])
        self.latest_cmd.linear.x = float(vx)
        self.latest_cmd.linear.y = 0.0
        age = 0.0 if self.last_path_update_time is None else (time.monotonic() - self.last_path_update_time)
        self.logger.debug(
            f"path target_vx={self.latest_cmd.linear.x:.3f} vyaw_ff={self.path_vyaw_ff:.3f} "
            f"backward={self.is_backward_segment} "
            f"path_age={age:.2f}s path_dt_ema={self.path_period_ema:.2f}s lookahead={step_idx}"
        )

    def destroy_node(self):
        self.logger.info("Destroying cmd_vel_control connection.")
        super().destroy_node()
        
def main(args=None):
    rclpy.init(args=args)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    node = CmdVelControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        
if __name__ == '__main__':
    main()
