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
        # === Heading-drift control: PI on heading drift ===
        # Each device has a constant open-loop yaw bias (measured ~0.05 rad/s on one
        # B2): with angular.z=0 it curves instead of driving straight.
        #   * The INTEGRAL term (yaw_bias_ki) learns this constant bias online and
        #     cancels it with zero steady-state error -- something pure P cannot do
        #     (pure P needs a standing heading error to hold the correction, which was
        #     the ~20 cm/10 m drift we measured in the field).
        #   * The PROPORTIONAL term (yaw_kp) provides DAMPING. Heading is the integral
        #     of yaw rate, so an integral-only controller on heading is an undamped
        #     oscillator (s^2 + ki = 0). P adds the s-term: s^2 + kp*s + ki = 0, with
        #     damping ratio zeta = kp / (2*sqrt(ki)). Without P the loop rings forever
        #     (verified in tool/planning_node_compare straight bench). DO NOT set kp=0.
        self.yaw_kp = 0.35             # proportional (damping) gain
        self.yaw_bias_ki = 0.15        # integral gain: rad/s of bias per (rad*s) drift
        self.yaw_bias_limit = 0.25     # clamp on the learned bias (rad/s)
        # Optional A-baseline: seeding the integral with the per-device field-measured
        # straight-line compensation (the angular.z that drives it straight) removes
        # the cold-start transient almost entirely (~50 cm over the first 10 m -> ~0 in
        # the straight bench). Leave 0 for pure self-learning; set per device to fix
        # the opening drift.
        self.yaw_bias_seed = 0.0
        self.drift_filter_tau = 0.15   # low-pass on drift before P/I
        # Integrate ONLY while the plan is going roughly straight; on turns the
        # feedforward is nonzero and the drift reflects intended turning, not bias.
        self.straight_ff_threshold = 0.05  # rad/s; |feedforward| below this == straight
        self._yaw_bias_est = self.yaw_bias_seed  # learned bias / integral state (rad/s)
        self._drift_lp = None          # low-passed drift state
        # Persist the self-learned bias across runs: load it as the seed on start, save
        # periodically + on shutdown. Removes the per-device cold-start drift with no
        # hand-entered yaw_bias_seed. Set env TINYNAV_CMDVEL_CALIB="" to disable (sim /
        # regression), or to another path to relocate it. Default matches the repo's
        # runtime scratch dir (map_node's --tinynav_db_path default).
        self.calib_path = os.environ.get("TINYNAV_CMDVEL_CALIB",
                                         os.path.join("tinynav_temp", "cmd_vel_calib.json"))
        self.calib_save_period_s = 10.0
        self._last_calib_save = time.monotonic()
        self._load_calib()
        # Static-friction compensation: very small vx often cannot move the robot.
        self.min_effective_linear_speed = 0.1
        # Yaw deadzone. Must stay BELOW the per-device yaw bias we need to cancel,
        # otherwise the bias correction (e.g. 0.05 rad/s) is snapped to 0 and the loop
        # can only deliver it by stuttering at the deadzone value -- the learned bias
        # then settles at ~2x the true bias and ~33 cm of drift persists (straight
        # bench). Field test confirmed 0.05 rad/s is executable, so 0.1 was too high.
        self.min_effective_angular_speed = 0.03
        self.linear_engage_threshold = 0.04
        self.fixed_reverse_speed = 0.2
        # Max segment heading change (rad) still treated as a straight reverse. Above
        # this, a backward-pointing lookahead chord is a forward U-turn, not reverse.
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

        # Yaw = planner FEEDFORWARD omega minus the LEARNED per-device yaw bias. The
        # bias is estimated by slowly integrating the heading drift (measured vs the
        # plan's intended heading) and fed forward; there is no per-tick P/D tracking.
        # Reference = orientation of the published trajectory pose nearest the control
        # center, so the planner decides how much to turn and B only removes drift.
        intended_yaw = self._path_intended_yaw()
        actual_yaw = self._actual_yaw()
        if intended_yaw is not None and actual_yaw is not None:
            drift = float(np.arctan2(np.sin(actual_yaw - intended_yaw),
                                     np.cos(actual_yaw - intended_yaw)))
            # Low-pass the drift first: intended_yaw comes from a sparse (~1 s spacing),
            # low-rate path, so it step-jumps when the nearest pose changes. Feeding
            # those steps straight into the integral would corrupt the bias estimate.
            if self._drift_lp is None:
                self._drift_lp = drift
            else:
                a = dt / (self.drift_filter_tau + dt)
                self._drift_lp += a * (drift - self._drift_lp)
            # Integrate (learn the bias) ONLY on straight, fresh, forward segments
            # (windup guard). The proportional term always acts -- it is the damping.
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

        # Faithfully reproduce the planner's instantaneous (v, omega) from the FIRST
        # published step (pose0 -> pose1), NOT a far ~0.8 m lookahead chord. The planner
        # builds every trajectory as a constant-curvature unicycle arc (vx in [0,0.3],
        # omega in +/-pi/3) plus a straight reverse vocabulary; the first published step
        # IS what to execute now. The old lookahead-chord velocity projected a sharp
        # forward U-turn backward (lookahead landed behind the robot) and misfired it as
        # straight reverse -- observed live, and in a bag a +154 deg trajectory was
        # already down to vx=+0.026, one step from flipping negative.
        T_robot_1 = self._pose_to_T(self.path.poses[0]) @ self.T_robot_to_camera
        T_robot_2 = self._pose_to_T(self.path.poses[1]) @ self.T_robot_to_camera
        T_robot_2_to_1 = np.linalg.inv(T_robot_1) @ T_robot_2
        disp = T_robot_2_to_1[:3, 3]
        dt = self.planner_dt * self.path_pose_stride        # one published step (~1.0 s)
        seg_yaw_change = float(R.from_matrix(T_robot_2_to_1[:3, :3]).as_rotvec()[2])

        # On an arc the robot's body-forward speed is the TANGENTIAL speed (arc length /
        # dt), not the chord's x-projection (which shrinks as the step curves). Sign it
        # by the forward direction so the straight reverse step still reads negative.
        fwd_sign = 1.0 if disp[0] >= 0.0 else -1.0
        raw_vx = fwd_sign * float(np.hypot(disp[0], disp[1])) / dt
        vyaw_seg = seg_yaw_change / dt

        # Reverse = planner's straight-back vocabulary (vx<0, omega=0). A forward arc
        # always keeps raw_vx>0 here (max one-step sweep ~60 deg < 90 deg), so the
        # heading-change guard is a safety net against any backward-projecting step.
        is_backward_segment = raw_vx < 0.0 and abs(seg_yaw_change) < self.reverse_max_yaw
        if is_backward_segment:
            vx = -self.fixed_reverse_speed
        else:
            vx = float(np.clip(raw_vx, 0.0, 0.5))

        # Feedforward yaw rate from the planned segment. The heading-drift PI is applied
        # per-tick in cmd_timer_callback against the planned intended heading.
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
