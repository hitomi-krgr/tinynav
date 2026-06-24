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
import os
import json

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
        # Camera sits this far ahead of the control center; heading must be referenced
        # at the control center or the short path lies behind the camera.
        self.cam_forward_offset = 1.0
        self.T_camera_to_control = self.T_robot_to_camera.copy()
        self.T_camera_to_control[2, 3] = -self.cam_forward_offset  # back along camera +z (=forward)
        self.last_path_time = 0.0
        self.pose = None
        self.path = None

        self.cmd_rate_hz = 12.0
        # Minima; actual stale thresholds are scaled by observed planner period.
        self.path_stale_slow_s = 0.35
        self.path_stale_stop_s = 0.8
        self.path_stale_slow_factor = 3.5
        self.path_stale_stop_factor = 5.0
        self.max_linear_acc = 0.6   # m/s^2
        self.max_angular_acc = 0.8  # rad/s^2
        # Match the planner's omega range; capping below it widens turn radius.
        self.max_angular_speed = float(np.pi / 3)  # rad/s, = planner omega max
        self.planner_dt = 0.1       # trajectory dt in planning_node
        self.path_pose_stride = 10  # planning_node publishes every 10th point (~1.0 s)
        self.path_period_ema = 0.12
        # Heading-drift control: PI on heading drift. I learns each device's constant
        # open-loop yaw bias (zero steady-state error); P provides damping (do not set 0).
        self.yaw_kp = 0.35             # proportional (damping) gain
        self.yaw_bias_ki = 0.15        # integral gain: rad/s of bias per (rad*s) drift
        self.yaw_bias_limit = 0.25     # clamp on the learned bias (rad/s)
        # Optional seed for the integral; leave 0 for pure self-learning.
        self.yaw_bias_seed = 0.0
        self.drift_filter_tau = 0.15   # low-pass on drift before P/I
        # Integrate only while the plan is roughly straight (feedforward near zero).
        self.straight_ff_threshold = 0.05  # rad/s; |feedforward| below this == straight
        self._yaw_bias_est = self.yaw_bias_seed  # learned bias / integral state (rad/s)
        self._drift_lp = None          # low-passed drift state
        # Persist the learned bias across runs. Env TINYNAV_CMDVEL_CALIB="" disables it.
        self.calib_path = os.environ.get("TINYNAV_CMDVEL_CALIB",
                                         os.path.join("tinynav_temp", "cmd_vel_calib.json"))
        self.calib_save_period_s = 10.0
        self._last_calib_save = time.monotonic()
        self._load_calib()
        # Static-friction compensation: very small vx often cannot move the robot.
        self.min_effective_linear_speed = 0.1
        # Yaw deadzone; must stay below the per-device yaw bias we cancel.
        self.min_effective_angular_speed = 0.03
        self.linear_engage_threshold = 0.04
        self.fixed_reverse_speed = 0.2
        # Max segment heading change still treated as a straight reverse (above = U-turn).
        self.reverse_max_yaw = 0.35

        self.latest_cmd = Twist()
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

    def _load_calib(self):
        """Seed the bias estimator from the last persisted value, if any."""
        if not self.calib_path:
            return
        try:
            with open(self.calib_path) as f:
                val = float(json.load(f)["yaw_bias"])
            self._yaw_bias_est = float(np.clip(val, -self.yaw_bias_limit, self.yaw_bias_limit))
            self.logger.info(
                f"Loaded yaw-bias calibration {self._yaw_bias_est:+.4f} rad/s from {self.calib_path}")
        except FileNotFoundError:
            pass
        except Exception as e:
            self.logger.warning(f"Could not load yaw-bias calibration ({self.calib_path}): {e}")

    def _save_calib(self, force=False):
        """Persist the learned bias (throttled, atomic)."""
        if not self.calib_path:
            return
        now = time.monotonic()
        if not force and (now - self._last_calib_save) < self.calib_save_period_s:
            return
        self._last_calib_save = now
        try:
            os.makedirs(os.path.dirname(self.calib_path) or ".", exist_ok=True)
            tmp = self.calib_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"yaw_bias": float(self._yaw_bias_est)}, f)
            os.replace(tmp, self.calib_path)
        except Exception as e:
            self.logger.warning(f"Could not save yaw-bias calibration ({self.calib_path}): {e}")

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

    def _actual_yaw(self):
        """World heading of the robot's measured forward axis (odometry)."""
        if self.pose is None:
            return None
        fwd = self._pose_to_T(self.pose.pose)[:3, :3] @ np.array([0.0, 0.0, 1.0])  # optical +z = forward
        return float(np.arctan2(fwd[1], fwd[0]))

    def _path_intended_yaw(self):
        """World heading the plan intends here: forward axis of the published trajectory
        pose nearest the control center. The reference for isolating open-loop drift."""
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

        # Yaw = planner feedforward omega minus the learned per-device yaw bias, where
        # the bias is integrated from the heading drift (measured vs plan-intended).
        intended_yaw = self._path_intended_yaw()
        actual_yaw = self._actual_yaw()
        if intended_yaw is not None and actual_yaw is not None:
            drift = float(np.arctan2(np.sin(actual_yaw - intended_yaw),
                                     np.cos(actual_yaw - intended_yaw)))
            # Low-pass the drift: intended_yaw comes from a sparse path and step-jumps
            # when the nearest pose changes, which would corrupt the integral.
            if self._drift_lp is None:
                self._drift_lp = drift
            else:
                a = dt / (self.drift_filter_tau + dt)
                self._drift_lp += a * (drift - self._drift_lp)
            # Learn the bias only on straight, fresh, forward segments (windup guard).
            straight = abs(self.path_vyaw_ff) < self.straight_ff_threshold
            if straight and age < stale_slow_s and not self.is_backward_segment:
                self._yaw_bias_est += self.yaw_bias_ki * self._drift_lp * dt
                self._yaw_bias_est = float(np.clip(self._yaw_bias_est,
                                                   -self.yaw_bias_limit, self.yaw_bias_limit))
                self._save_calib()   # throttled; persists the live estimate
            vyaw = self.path_vyaw_ff - (self.yaw_kp * self._drift_lp + self._yaw_bias_est)
        else:
            # No path/pose yet: feedforward minus the bias learned so far.
            self._drift_lp = None
            vyaw = self.path_vyaw_ff - self._yaw_bias_est
        target_cmd.angular.z = float(np.clip(vyaw, -self.max_angular_speed, self.max_angular_speed))

        if age > stale_stop_s:
            target_cmd.linear.x = 0.0
            target_cmd.angular.z = 0.0
        elif age > stale_slow_s:
            target_cmd.linear.x *= 0.3
            target_cmd.angular.z *= 0.5

        out = Twist()
        out.linear.y = 0.0

        # Reverse is a fixed-speed straight-back vocabulary; pass it through unsmoothed.
        if target_cmd.linear.x < 0.0:
            out.linear.x = target_cmd.linear.x
            out.angular.z = 0.0
            self.cmd_pub.publish(out)
            self.prev_cmd = out
            return

        # Forward/turning commands get acceleration limiting and minimum-speed locks.
        max_dv = self.max_linear_acc * dt
        # Just left reverse: don't let acceleration limiting leak another reverse command.
        prev_linear_x = 0.0 if self.prev_cmd.linear.x < 0.0 else self.prev_cmd.linear.x
        out.linear.x = self._clamp_step(target_cmd.linear.x, prev_linear_x, max_dv)
        # Don't acceleration-limit yaw; the turn rate is already decided upstream.
        out.angular.z = float(np.clip(target_cmd.angular.z, -self.max_angular_speed, self.max_angular_speed))

        # Tiny non-zero forward speeds aren't executable: creep at +min for any positive
        # target (else the robot freezes and deadlocks); non-positive target decays to 0.
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

        # Reproduce the planner's instantaneous (v, omega) from the first published step
        # (pose0 -> pose1), not a far lookahead chord which can misfire as reverse.
        T_robot_1 = self._pose_to_T(self.path.poses[0]) @ self.T_robot_to_camera
        T_robot_2 = self._pose_to_T(self.path.poses[1]) @ self.T_robot_to_camera
        T_robot_2_to_1 = np.linalg.inv(T_robot_1) @ T_robot_2
        disp = T_robot_2_to_1[:3, 3]
        dt = self.planner_dt * self.path_pose_stride        # one published step (~1.0 s)
        seg_yaw_change = float(R.from_matrix(T_robot_2_to_1[:3, :3]).as_rotvec()[2])

        # Body-forward speed is the tangential (arc-length) speed, not the chord's
        # x-projection. Signed by forward direction so a reverse step reads negative.
        fwd_sign = 1.0 if disp[0] >= 0.0 else -1.0
        raw_vx = fwd_sign * float(np.hypot(disp[0], disp[1])) / dt
        vyaw_seg = seg_yaw_change / dt

        # Reverse = straight-back vocabulary (vx<0, omega=0); the yaw guard rejects
        # backward-projecting forward arcs.
        is_backward_segment = raw_vx < 0.0 and abs(seg_yaw_change) < self.reverse_max_yaw
        if is_backward_segment:
            vx = -self.fixed_reverse_speed
        else:
            vx = float(np.clip(raw_vx, 0.0, 0.4))
            # Preserve turn radius (vx/omega) when omega exceeds the cap: scale vx by the
            # same ratio instead of just clipping omega (which would widen the radius).
            if abs(vyaw_seg) > self.max_angular_speed:
                vx *= self.max_angular_speed / abs(vyaw_seg)
                vyaw_seg = float(np.sign(vyaw_seg) * self.max_angular_speed)

        # Feedforward yaw rate; the heading-drift PI is applied per-tick in the timer.
        self.is_backward_segment = is_backward_segment
        self.path_vyaw_ff = 0.0 if is_backward_segment else vyaw_seg
        self.latest_cmd.linear.x = float(vx)
        self.latest_cmd.linear.y = 0.0
        age = 0.0 if self.last_path_update_time is None else (time.monotonic() - self.last_path_update_time)
        self.logger.debug(
            f"path target_vx={self.latest_cmd.linear.x:.3f} vyaw_ff={self.path_vyaw_ff:.3f} "
            f"backward={self.is_backward_segment} seg_yaw={seg_yaw_change:+.3f} "
            f"path_age={age:.2f}s path_dt_ema={self.path_period_ema:.2f}s"
        )

    def destroy_node(self):
        self._save_calib(force=True)
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
