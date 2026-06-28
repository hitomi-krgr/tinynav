#!/usr/bin/env python3
"""Node-in-the-loop comparison of planning_node variants on a real map.

Drives the *real* planning-node + cmd_vel_control files end-to-end on a pre-built
tinynav map: loads the map's occupancy + SDF, runs the SAME SDF-A* the map_node
uses to chain a global path through a sequence of POIs, renders synthetic depth
from the map occupancy, spins each planning node + the real controller in the
loop, integrates the robot from /cmd_vel, and overlays every node's executed
trajectory on the map.

    REQUIRES A ROS 2 (humble) ENVIRONMENT — the nodes import rclpy / cv_bridge at
    module load. Run inside a container, e.g.:

        docker run --rm -it -v "$PWD":/ws -w /ws ros:humble bash
        apt-get update && apt-get install -y python3-pip ros-humble-cv-bridge
        pip3 install "numpy<2" scipy numba "opencv-python-headless==4.9.0.80" \
            matplotlib codetiming fufpy async_lru
        source /opt/ros/humble/setup.bash
        export PYTHONPATH=/ws:$PYTHONPATH
        python3 tool/planning_node_compare.py \
            tinynav/core/planning_node.py \
            tinynav/core/planning_node_turn.py \
            tinynav/core/planning_node_localtarget.py \
            --map tinynav_db/maps/<map_dir> --pois "home,printer"

Plot lands in <out>/<route>.png (all nodes overlaid on the map).
"""
import argparse
import heapq
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from cv_bridge import CvBridge
import tf2_ros

from tinynav.core.math_utils import matrix_to_quat

_HERE = os.path.dirname(os.path.abspath(__file__))


# --- camera model (wide-FOV depth, optical frame z-fwd / x-right / y-down) ---
IMG_W, IMG_H = 160, 120
HFOV_DEG = 90.0
FX = (IMG_W / 2.0) / np.tan(np.deg2rad(HFOV_DEG) / 2.0)
FY = FX
CX, CY = IMG_W / 2.0, IMG_H / 2.0
# Occupied cells are treated as full-height columns spanning this z band so they
# fall inside the node's obstacle z-band; camera sits near world z = 0.
WALL_Z0, WALL_Z1 = -1.0, 1.0

# --- closed-loop drive --------------------------------------------------------
CAM_FORWARD_OFFSET = 1.0    # B2: camera is 1.0m ahead of the control center
GOAL_EPS_M = 0.40
WARMUP_CYCLES = 6           # let the occupancy map fill in before moving
SPIN_TRIES = 20
SPIN_DT = 0.02
CTRL_DT = 1.0 / 12.0        # controller tick (matches cmd_rate_hz)
PLAN_EVERY_TICKS = 3        # republish full sensor bundle (replan) every N ticks
CMDVEL_MAX_T = 120.0        # sim-seconds budget per route
TARGET_LOOKAHEAD_M = 2.5    # lookahead distance along the global path (map_node parity)


def _precompute_rays():
    """Unit ray directions in the camera optical frame, one per pixel."""
    uu, vv = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
    dirs = np.stack([(uu - CX) / FX, (vv - CY) / FY, np.ones_like(uu)], axis=-1).reshape(-1, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs.astype(np.float64)


RAYS_CAM = _precompute_rays()


def footprint_collides(cx, cy, yaw, fp, occ2d, ox, oy, res):
    """True if the robot footprint (control-center frame) overlaps an occupied cell."""
    front_len, rear_len, half_w = fp
    c, s = np.cos(yaw), np.sin(yaw)
    nx, ny = occ2d.shape
    step = res * 0.75
    bx = -rear_len
    while bx <= front_len + 1e-6:
        by = -half_w
        while by <= half_w + 1e-6:
            wx = cx + bx * c - by * s
            wy = cy + bx * s + by * c
            ix = int((wx - ox) / res)
            iy = int((wy - oy) / res)
            if 0 <= ix < nx and 0 <= iy < ny and occ2d[ix, iy]:
                return True
            by += step
        bx += step
    return False


def points_in_occ(path_xy, occ2d, ox, oy, res):
    """How many global/published path points land in occupied cells (ground truth)."""
    if path_xy is None or len(path_xy) == 0:
        return 0
    nx, ny = occ2d.shape
    n = 0
    for x, y in path_xy:
        ix, iy = int((x - ox) / res), int((y - oy) / res)
        if 0 <= ix < nx and 0 <= iy < ny and occ2d[ix, iy]:
            n += 1
    return n


def yaw_to_R_wc(yaw):
    """Camera->world rotation for a planar heading. Optical z=forward, y=down."""
    c, s = np.cos(yaw), np.sin(yaw)
    z_axis = np.array([c, s, 0.0])        # forward
    y_axis = np.array([0.0, 0.0, -1.0])   # down
    x_axis = np.cross(y_axis, z_axis)     # right = y x z
    return np.column_stack([x_axis, y_axis, z_axis])


def render_depth_grid(cam_pos, R_wc, occ2d, ox, oy, res, max_range=8.0, step=0.15):
    """Render depth against a 2D occupancy mask (occupied cells = full-height
    columns spanning [WALL_Z0, WALL_Z1]). Ray-march each pixel; first hit wins.
    Matches the node's pinhole unprojection px=(u-cx)d/fx, py=(v-cy)d/fy, pz=d."""
    P = RAYS_CAM.shape[0]
    dirs_w = RAYS_CAM @ R_wc.T
    cos_ang = dirs_w @ R_wc[:, 2]          # depth(z) = world_t * cos_ang
    nx, ny = occ2d.shape
    depth = np.zeros(P)
    hit = np.zeros(P, dtype=bool)
    t = step
    for _ in range(int(max_range / step)):
        pts = cam_pos[None, :] + dirs_w * t
        within_z = (pts[:, 2] >= WALL_Z0) & (pts[:, 2] <= WALL_Z1)
        ix = ((pts[:, 0] - ox) / res).astype(int)
        iy = ((pts[:, 1] - oy) / res).astype(int)
        valid = within_z & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & ~hit
        cells = np.zeros(P, dtype=bool)
        cells[valid] = occ2d[ix[valid], iy[valid]]
        newhit = valid & cells
        depth[newhit] = t * cos_ang[newhit]
        hit |= newhit
        t += step
    depth = np.where(hit & (depth > 0) & (depth < max_range * 1.2), depth, 0.0)
    return depth.reshape(IMG_H, IMG_W).astype(np.float32)


# --- map + SDF-A* (copied from map_node.search_within_sdf_map) ---------------
@dataclass
class Scene:
    name: str
    global_path_pts: list = field(default_factory=list)  # (x, y) waypoints (world)
    robot: tuple = (0.0, 0.0, 0.0)                        # (x, y, yaw)
    target: Optional[np.ndarray] = None
    occ2d: Optional[np.ndarray] = None                    # map occupancy (for depth/plot/collision)
    occ_ox: float = 0.0
    occ_oy: float = 0.0
    occ_res: float = 0.1


def _map_heuristic(a, b, res):
    va, vb = np.array(a), np.array(b)
    return float(np.linalg.norm((va - vb) * res) + 20 * abs(va[2] - vb[2]) * res)


def _map_reconstruct(parent, current):
    path = []
    while current in parent:
        path.append(current)
        if current == parent[current]:
            break
        current = parent[current]
    return path[::-1]


def _map_astar(start, goal, sdf, occ, res):
    bins = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    def qi(v):
        for i, t in enumerate(bins):
            if v < t:
                return i
        return len(bins)

    H = [[] for _ in range(len(bins) + 1)]
    S = [set() for _ in range(len(bins) + 1)]
    si = qi(float(sdf[start]))
    heapq.heappush(H[si], (_map_heuristic(start, goal, res), start))
    S[si].add(start)
    parent = {start: start}
    visited = set()
    while True:
        qx = -1
        for i, q in enumerate(H):
            if q:
                qx = i
                break
        if qx == -1:
            break
        _, cur = heapq.heappop(H[qx])
        S[qx].discard(cur)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            return _map_reconstruct(parent, cur)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    nb = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                    if not (0 <= nb[0] < sdf.shape[0] and 0 <= nb[1] < sdf.shape[1] and 0 <= nb[2] < sdf.shape[2]):
                        continue
                    if nb in visited or occ[nb] == 2:
                        continue
                    ni = qi(float(sdf[nb]))
                    if nb in S[ni]:
                        continue
                    S[ni].add(nb)
                    heapq.heappush(H[ni], (_map_heuristic(nb, goal, res), nb))
                    if nb not in parent:
                        parent[nb] = cur
    return []


def load_map(map_dir):
    occ = np.load(os.path.join(map_dir, "occupancy_grid.npy"))
    sdf = np.load(os.path.join(map_dir, "sdf_map.npy"))
    meta = np.load(os.path.join(map_dir, "occupancy_meta.npy"))
    with open(os.path.join(map_dir, "pois.json")) as f:
        pois_raw = json.load(f)
    pois = {v["name"]: np.array(v["position"], dtype=np.float64) for v in pois_raw.values()}
    oz, res = float(meta[2]), float(meta[3])
    # Collapse occupancy only over the robot's working z-band, NOT all z. Collapsing
    # every layer turns floor / ceiling / overhead structure into 2D walls and makes
    # corners far tighter than the robot actually sees -> sim wedges where real drives.
    # The band is centred on the SDF-seed height (the recorded mapping pose z, i.e. where
    # the robot physically was during collection), not the POI height (targets may sit high).
    work_z = oz
    pp = os.path.join(map_dir, "poses.npy")
    if os.path.exists(pp):
        pd = np.load(pp, allow_pickle=True).item()
        if pd:
            work_z = float(np.median([np.asarray(v)[2, 3] for v in pd.values()]))
    z_world = oz + (np.arange(occ.shape[2]) + 0.5) * res
    band = (z_world >= work_z - 0.4) & (z_world <= work_z + 0.4)
    occ2d = (occ[:, :, band] == 2).any(axis=2) if band.any() else (occ == 2).any(axis=2)
    return {
        "occ": occ, "sdf": sdf,
        "ox": float(meta[0]), "oy": float(meta[1]), "oz": oz, "res": res,
        "pois": pois,
        "occ2d": occ2d,
    }


def load_start_from_poses(map_dir):
    """Earliest-timestamp pose translation from the recorded mapping trajectory.
    poses.npy is a pickled {timestamp_ns: 4x4} dict; the first frame is where the
    robot began mapping -> a realistic, on-floor start (unlike a POI, which is a
    target that may sit against a wall). Returns (x, y, z) or None."""
    p = os.path.join(map_dir, "poses.npy")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True).item()
    if not d:
        return None
    pose0 = np.asarray(d[min(d.keys())])
    return tuple(float(v) for v in pose0[:3, 3])


def repair_path(path_w, occ2d, ox, oy, res, push=0.2):
    """Push global-path points that sit in (or within `push` of) an obstacle out to
    the nearest cell with >= push clearance."""
    from scipy.ndimage import distance_transform_edt
    clear = distance_transform_edt(~occ2d) * res
    nx, ny = occ2d.shape
    R = int(push / res) + 3
    out = []
    for (x, y) in path_w:
        ix, iy = int((x - ox) / res), int((y - oy) / res)
        if 0 <= ix < nx and 0 <= iy < ny and clear[ix, iy] >= push:
            out.append((x, y)); continue
        best, bd = None, 1e18
        for a in range(-R, R + 1):
            for b in range(-R, R + 1):
                xi, yi = ix + a, iy + b
                if 0 <= xi < nx and 0 <= yi < ny and clear[xi, yi] >= push and a * a + b * b < bd:
                    bd, best = a * a + b * b, (xi, yi)
        out.append((best[0] * res + ox, best[1] * res + oy) if best else (x, y))
    return out


def map_scene(mapd, names, path_stride=3, start_w=None):
    """Build a Scene from waypoints: chain SDF-A* between consecutive waypoints
    into one global path; obstacles = map occupied cells. With start_w given, the
    route is start_w -> POIs (so a single-POI map is still a runnable route and
    the start is the recorded mapping origin instead of a POI)."""
    pois = mapd["pois"]
    for nm in names:
        if nm not in pois:
            raise SystemExit(f"POI '{nm}' not found. available: {sorted(pois)}")
    ox, oy, oz, res = mapd["ox"], mapd["oy"], mapd["oz"], mapd["res"]
    occ, sdf = mapd["occ"], mapd["sdf"]

    def w2i(p):
        return (int((p[0] - ox) / res), int((p[1] - oy) / res), int((p[2] - oz) / res))

    waypts = ([np.asarray(start_w, dtype=np.float64)] if start_w is not None else []) \
        + [np.asarray(pois[n], dtype=np.float64) for n in names]
    if len(waypts) < 2:
        raise SystemExit("need >=2 waypoints (use --start-from-poses for single-POI maps)")

    idx_path = []
    for a, b in zip(waypts[:-1], waypts[1:]):
        seg = _map_astar(w2i(a), w2i(b), sdf, occ, res)
        if not seg:
            raise SystemExit(f"no global path ({a[:2]} -> {b[:2]})")
        idx_path.extend(seg if not idx_path else seg[1:])
    path_w = [(i * res + ox, j * res + oy) for (i, j, _k) in idx_path[::path_stride]]
    end_w = (idx_path[-1][0] * res + ox, idx_path[-1][1] * res + oy)
    if path_w[-1] != end_w:
        path_w.append(end_w)
    sw, gw = waypts[0], waypts[-1]
    occ2d = mapd["occ2d"]
    # Push global-path points out of obstacles to >= 0.2 m clearance.
    path_w = repair_path(path_w, occ2d, ox, oy, res, push=0.2)
    # Start facing along the (repaired) global path's first segment.
    yaw0 = float(np.arctan2(path_w[1][1] - path_w[0][1], path_w[1][0] - path_w[0][0])) if len(path_w) >= 2 else 0.0
    tag = ("start_" if start_w is not None else "") + "_".join(names)
    return Scene(name="map_" + tag,
                 global_path_pts=path_w,
                 robot=(float(sw[0]), float(sw[1]), yaw0),
                 target=np.array([gw[0], gw[1], gw[2]], dtype=np.float64),
                 occ2d=occ2d, occ_ox=ox, occ_oy=oy, occ_res=res)


class ProgressTracker:
    """Monotonic path follower. On a loop tour the same (x,y) appears at both ends,
    so a global-nearest lookahead snaps back to the start; this only ever searches
    a window AHEAD of the committed index, so progress moves forward."""

    def __init__(self, path_xy, window=10):
        self.path = path_xy
        self.win = window
        self.prog = 0

    def update(self, cx, cy):
        lo = self.prog
        hi = min(len(self.path), lo + self.win)
        seg = self.path[lo:hi]
        self.prog = lo + int(np.argmin(np.linalg.norm(seg - np.array([cx, cy]), axis=1)))
        return self.prog

    def target(self, cx, cy):
        """Lookahead point ~TARGET_LOOKAHEAD_M ahead of current progress."""
        self.update(cx, cy)
        acc, k = 0.0, self.prog
        while k < len(self.path) - 1 and acc < TARGET_LOOKAHEAD_M:
            acc += float(np.linalg.norm(self.path[k + 1] - self.path[k]))
            k += 1
        return np.array([self.path[k][0], self.path[k][1], 0.0])

    def reached(self, cx, cy, final_goal):
        return (self.prog >= len(self.path) - 2
                and float(np.linalg.norm(final_goal - np.array([cx, cy]))) < GOAL_EPS_M)


class DriverNode(Node):
    """Publishes synthetic sensors from the map and captures the planner path / cmd_vel."""

    def __init__(self):
        super().__init__("pn_compare_driver")
        self.bridge = CvBridge()
        self.depth_pub = self.create_publisher(Image, "/slam/depth", 10)
        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry_visual", 10)   # planner
        self.odom_cam_pub = self.create_publisher(Odometry, "/slam/odometry", 10)        # controller
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera/infra2/camera_info", 10)
        self.target_pub = self.create_publisher(Odometry, "/control/target_pose", 10)
        self.gplan_pub = self.create_publisher(Path, "/mapping/global_plan", 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.create_subscription(Path, "/planning/trajectory_path", self._on_path, 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.latest_path = None
        self.latest_cmd = (0.0, 0.0)
        self._occ = None   # (occ2d, ox, oy, res)
        self._seq = 0

    def _on_path(self, msg: Path):
        self.latest_path = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])

    def _on_cmd(self, msg: Twist):
        self.latest_cmd = (float(msg.linear.x), float(msg.angular.z))

    def _stamp(self):
        self._seq += 1
        return Time(nanoseconds=self._seq * 100_000_000).to_msg()  # 0.1s increments

    @staticmethod
    def _odom_msg(cam_pos, R_wc, stamp):
        q = matrix_to_quat(R_wc)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "camera"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = cam_pos
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = float(q[0]), float(q[1])
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = float(q[2]), float(q[3])
        return odom

    def publish_camera_odom(self, cam_pos, R_wc):
        self.odom_cam_pub.publish(self._odom_msg(cam_pos, R_wc, self._stamp()))

    def publish_camera_info(self):
        msg = CameraInfo()
        msg.header.frame_id = "camera"
        msg.width, msg.height = IMG_W, IMG_H
        msg.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
        msg.p = [FX, 0.0, CX, -FX * 0.05, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(msg)

    def publish_world(self, cam_pos, R_wc, target_xy, global_path):
        stamp = self._stamp()
        depth = render_depth_grid(cam_pos, R_wc, *self._occ)
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = "camera"

        odom = self._odom_msg(cam_pos, R_wc, stamp)

        tgt = Odometry()
        tgt.header.stamp = stamp
        tgt.header.frame_id = "world"
        tgt.pose.pose.position.x, tgt.pose.pose.position.y = float(target_xy[0]), float(target_xy[1])
        tgt.pose.pose.orientation.w = 1.0

        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = "map"
        for x, y in global_path:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "world"
        tf.child_frame_id = "map"
        tf.transform.rotation.w = 1.0

        # depth + odom must share a stamp for the node's TimeSynchronizer.
        self.depth_pub.publish(depth_msg)
        self.odom_pub.publish(odom)
        self.odom_cam_pub.publish(odom)
        self.target_pub.publish(tgt)
        self.gplan_pub.publish(path_msg)
        self.tf_broadcaster.sendTransform(tf)


def load_planning_node_class(path):
    name = "pn_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.PlanningNode


# Shared virtual clock so the controller's time.monotonic()-based dt / path-staleness
# logic runs deterministically in sim time instead of wall-clock.
SIM_CLOCK = [0.0]

# Field-measured device open-loop yaw bias (rad/s), injected into the integrated
# robot motion so the controller's online bias estimator (B) has something to learn.
# 0 => ideal device. On the measured B2 the bias was ~0.05 rad/s.
DEVICE_YAW_BIAS = float(os.environ.get("PNC_DEVICE_YAW_BIAS", "0.0"))


def load_controller(path):
    # Disable the on-disk yaw-bias persistence so sim runs are deterministic and never
    # load a stale calibration or write into the repo.
    os.environ["TINYNAV_CMDVEL_CALIB"] = ""
    spec = importlib.util.spec_from_file_location("cmdvel_ctrl", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cmdvel_ctrl"] = mod
    spec.loader.exec_module(mod)
    mod.time = types.SimpleNamespace(monotonic=lambda: SIM_CLOCK[0])
    return mod.CmdVelControlNode


def reset_node_state(node):
    node.occupancy_grid = np.zeros(node.grid_shape)
    node.origin = np.array(node.grid_shape) * node.resolution / -2.0
    node.last_T = None
    node.last_param = (0.0, 0.0)
    node.target_pose = None
    node.smoothed_velocity = 0.0
    for attr in ("_last_local_target", "global_path_odom"):
        if hasattr(node, attr):
            setattr(node, attr, None)


def reset_controller_state(ctrl):
    # cmd_vel_control gates all output on /nav/active (added in #138, after this
    # harness was written). Nothing publishes that topic in sim, so force the
    # controller active or it returns zero cmd forever and the robot never moves.
    ctrl._nav_active = True
    ctrl.pose = None
    ctrl.path = None
    ctrl.latest_cmd = Twist()
    ctrl.prev_cmd = Twist()
    ctrl.path_vyaw_ff = 0.0
    ctrl.is_backward_segment = False
    ctrl._yaw_bias_est = ctrl.yaw_bias_seed
    ctrl._drift_lp = None
    ctrl.last_path_update_time = None
    ctrl.path_period_ema = 0.12
    ctrl.last_cmd_pub_time = 0.0


def run_route(node, ctrl, driver, executor, scene, fp):
    """Closed-loop rollout (real cmd_vel_control in the loop). Returns
    {traj, collided}: control-center xy track and the collision point (or None)."""
    reset_node_state(node)
    reset_controller_state(ctrl)
    driver._occ = (scene.occ2d, scene.occ_ox, scene.occ_oy, scene.occ_res)
    driver.latest_path = None
    driver.latest_cmd = (0.0, 0.0)
    global_path = np.asarray(scene.global_path_pts, dtype=np.float64)
    final_goal = global_path[-1]
    tracker = ProgressTracker(global_path)

    # --- openness prior (emulates map_node's static obstacle-distance hint) ---
    # When PNC_OPENNESS_SAFETY is set, modulate node.robot.safety_radius per-frame
    # from the static obstacle EDT at the robot: open -> generous, narrow -> tight.
    _openness = None
    if os.environ.get("PNC_OPENNESS_SAFETY"):
        from scipy.ndimage import distance_transform_edt as _edt
        _openness = _edt(~scene.occ2d) * scene.occ_res
        _base_safety = float(node.robot.safety_radius)

    def _openness_at(px, py):
        ix = int((px - scene.occ_ox) / scene.occ_res)
        iy = int((py - scene.occ_oy) / scene.occ_res)
        if 0 <= ix < _openness.shape[0] and 0 <= iy < _openness.shape[1]:
            return float(_openness[ix, iy])
        return None

    def _localtarget_and_openness(px, py, lookahead_max=TARGET_LOOKAHEAD_M,
                                  min_lookahead=1.0, turn_thresh=np.deg2rad(45.0), smooth_m=0.4):
        # Mirror of map_node: local target = furthest path point reachable before the
        # heading turns past turn_thresh (a corner) or lookahead_max, never closer than
        # min_lookahead. openness = MIN obstacle-ESDF over robot -> local-target.
        gp = global_path[:, :2]
        n = len(gp)
        i0 = int(np.argmin(np.linalg.norm(gp - np.array([px, py]), axis=1)))
        cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))))

        def sdir(i):
            j = i
            while j < n - 1 and (cum[j] - cum[i]) < smooth_m:
                j += 1
            dvec = gp[j] - gp[i]; L = float(np.linalg.norm(dvec))
            return dvec / L if L > 1e-6 else None

        entry = sdir(i0)
        li = i0
        for k in range(i0 + 1, n):
            if cum[k] - cum[i0] >= lookahead_max:
                li = k; break
            dk = sdir(k)
            if (cum[k] - cum[i0]) >= min_lookahead and entry is not None and dk is not None:
                turn = abs(np.arctan2(dk[0] * entry[1] - dk[1] * entry[0], float(dk @ entry)))
                if turn >= turn_thresh:
                    li = k; break
            li = k
        vals = [v for v in (_openness_at(p[0], p[1]) for p in gp[i0:li + 1]) if v is not None]
        return gp[li], (min(vals) if vals else None)

    cx, cy, yaw = scene.robot
    samples = [(cx, cy)]
    collisions = []          # all collision points along the route (do not stop on hit)
    _in_collision = False

    def cam_pose():
        R = yaw_to_R_wc(yaw)
        return np.array([cx, cy, 0.0]) + R[:, 2] * CAM_FORWARD_OFFSET, R

    # Warmup: build the map + seed a first path while stationary.
    for _ in range(WARMUP_CYCLES):
        cam_pos, R = cam_pose()
        SIM_CLOCK[0] += CTRL_DT
        driver.publish_world(cam_pos, R, tracker.target(cx, cy)[:2], global_path)
        for _ in range(SPIN_TRIES):
            executor.spin_once(timeout_sec=SPIN_DT)

    for k in range(int(CMDVEL_MAX_T / CTRL_DT)):
        if tracker.reached(cx, cy, final_goal):
            break
        cam_pos, R = cam_pose()
        SIM_CLOCK[0] += CTRL_DT

        # Replan every few ticks; clear latest_path first and wait for a FRESH path
        # so the controller never chases a stale one.
        is_plan_tick = (k % PLAN_EVERY_TICKS == 0)
        if is_plan_tick:
            target_xy = tracker.target(cx, cy)[:2]
            if _openness is not None:
                lt, o = _localtarget_and_openness(cx, cy)
                target_xy = lt  # publish localtarget in place of the fixed lookahead
                if o is not None:
                    b = _base_safety
                    node.robot.safety_radius = min(b, max(b * 0.75, (o - node.robot.width) * 0.75))
            driver.latest_path = None
            driver.publish_world(cam_pos, R, target_xy, global_path)
        else:
            driver.publish_camera_odom(cam_pos, R)
        for _ in range(SPIN_TRIES):
            executor.spin_once(timeout_sec=SPIN_DT)
            if ctrl.pose is not None and (not is_plan_tick or driver.latest_path is not None):
                break

        # Tick the controller manually (its ROS timer is cancelled) and integrate.
        ctrl.cmd_timer_callback()
        executor.spin_once(timeout_sec=SPIN_DT)   # deliver /cmd_vel to the driver
        vx, wz = driver.latest_cmd
        # The real device rotates at commanded wz PLUS its intrinsic open-loop bias.
        yaw += (wz + DEVICE_YAW_BIAS) * CTRL_DT
        cx += vx * np.cos(yaw) * CTRL_DT
        cy += vx * np.sin(yaw) * CTRL_DT
        samples.append((cx, cy))

        # Collision: footprint overlaps a real occupied cell. Record it (rising edge)
        # but KEEP GOING -- we want the full route and every collision, not a stop.
        hit = footprint_collides(cx, cy, yaw, fp, scene.occ2d, scene.occ_ox, scene.occ_oy, scene.occ_res)
        if hit and not _in_collision:
            collisions.append((cx, cy))
            if os.environ.get("PNC_DEBUG"):
                p_in = points_in_occ(driver.latest_path, scene.occ2d, scene.occ_ox, scene.occ_oy, scene.occ_res)
                print(f"    COLLISION #{len(collisions)} @({cx:.2f},{cy:.2f}) step={k} | "
                      f"published-path pts in wall={p_in}/{0 if driver.latest_path is None else len(driver.latest_path)}",
                      flush=True)
        _in_collision = hit

        if os.environ.get("PNC_DEBUG") and k % 24 == 0:
            p_in = points_in_occ(driver.latest_path, scene.occ2d, scene.occ_ox, scene.occ_oy, scene.occ_res)
            herr = ctrl._live_heading_err() if hasattr(ctrl, "_live_heading_err") else None
            print(f"    t={SIM_CLOCK[0]:.1f} prog={tracker.prog}/{len(global_path)} "
                  f"pos=({cx:.2f},{cy:.2f}) yaw={yaw:.2f} vx={vx:.3f} wz={wz:.3f} "
                  f"herr={herr} pathInWall={p_in}", flush=True)

    return {"traj": np.array(samples), "collisions": collisions,
            "collided": collisions[0] if collisions else None}


def lateral_deviation(traj, global_path):
    """Max & final perpendicular distance [m] of the executed track from the
    global-path polyline (nearest-vertex approximation)."""
    gp = np.asarray(global_path)[:, :2]
    d = np.array([np.min(np.linalg.norm(gp - p, axis=1)) for p in np.asarray(traj)[:, :2]])
    return float(d.max()), float(d[-1])


COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
STYLES = ["-", ":", "--", "-.", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]


def plot_route(scene, results, labels, out_path):
    fig, ax = plt.subplots(figsize=(11, 10))
    occ2d = scene.occ2d
    nx, ny = occ2d.shape
    extent = [scene.occ_ox, scene.occ_ox + nx * scene.occ_res,
              scene.occ_oy, scene.occ_oy + ny * scene.occ_res]
    ax.imshow(occ2d.T, extent=extent, origin="lower", cmap="Greys", alpha=0.85)

    gp = np.asarray(scene.global_path_pts)
    ax.plot(gp[:, 0], gp[:, 1], "y--", lw=1.4, alpha=0.85, label="global_path (SDF-A*)")

    rx, ry, ryaw = scene.robot
    ax.plot(rx, ry, "bs", ms=12, label="start")
    ax.arrow(rx, ry, 0.5 * np.cos(ryaw), 0.5 * np.sin(ryaw),
             head_width=0.25, head_length=0.18, fc="b", ec="b")
    ax.plot(scene.target[0], scene.target[1], "kX", ms=14, mew=2, label="goal")

    for i, (label, res) in enumerate(zip(labels, results)):
        c, st = COLORS[i % len(COLORS)], STYLES[i % len(STYLES)]
        traj = res["traj"]
        cols = res.get("collisions", [])
        tag = f"  {len(cols)} hits" if cols else ""
        ax.plot(traj[:, 0], traj[:, 1], color=c, linestyle=st, lw=2.6, alpha=0.9,
                label=f"{label} ({len(traj) - 1} steps){tag}")
        ax.plot(traj[-1, 0], traj[-1, 1], marker=".", color=c, ms=14)
        for (hx, hy) in cols:
            ax.plot(hx, hy, "X", color=c, ms=14, mew=2)

    ax.set_title(scene.name)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="planning_node .py files to compare")
    ap.add_argument("--map", required=True, help="pre-built map dir (occupancy_grid/sdf_map/pois.json)")
    ap.add_argument("--pois", required=True,
                    help="POI route(s): waypoint sequence 'home,boss,sit,printer,home'. "
                         "Use ';' for several independent routes.")
    ap.add_argument("--out", default=os.path.join(_HERE, "out_map"))
    ap.add_argument("--controller-path",
                    default=os.path.join(_HERE, "..", "tinynav", "platforms", "cmd_vel_control.py"))
    ap.add_argument("--max-time", type=float, default=None, help="sim-seconds budget per route")
    ap.add_argument("--robot", choices=["go2", "b2"], default="go2",
                    help="override the robot config in the loaded nodes (deployment files unchanged)")
    ap.add_argument("--start-from-poses", action="store_true",
                    help="use poses.npy[earliest] as the route start (route = start -> POIs); "
                         "lets single-POI maps run and starts from the recorded mapping origin")
    args = ap.parse_args()

    global CMDVEL_MAX_T, CAM_FORWARD_OFFSET
    if args.max_time is not None:
        CMDVEL_MAX_T = args.max_time

    os.makedirs(args.out, exist_ok=True)
    mapd = load_map(args.map)
    print(f"map: {args.map} | POIs: {sorted(mapd['pois'])}", flush=True)
    start_w = load_start_from_poses(args.map) if args.start_from_poses else None
    if args.start_from_poses and start_w is None:
        raise SystemExit("--start-from-poses set but no usable poses.npy in the map dir")
    routes = []
    for route in args.pois.split(";"):
        names = [x.strip() for x in route.split(",") if x.strip()]
        if not names or (start_w is None and len(names) < 2):
            raise SystemExit("each --pois route needs >=2 POIs (or pass --start-from-poses)")
        routes.append(map_scene(mapd, names, start_w=start_w))

    labels = [os.path.splitext(os.path.basename(p))[0] for p in args.paths]
    results = [[None] * len(args.paths) for _ in routes]

    rclpy.init()
    try:
        driver = DriverNode()
        for fi, path in enumerate(args.paths):
            print(f"\n=== {labels[fi]} ({path}) ===", flush=True)
            node = load_planning_node_class(path)()
            # Override the robot config in-sim (deployment files keep B2_CONFIG).
            node_mod = sys.modules[type(node).__module__]
            cfg = getattr(node_mod, "GO2_CONFIG" if args.robot == "go2" else "B2_CONFIG")
            node.robot = cfg
            CAM_FORWARD_OFFSET = float(cfg.camera_x - cfg.control_x)

            ctrl = load_controller(args.controller_path)()
            # Closed-loop controllers reference the control center; main's open-loop
            # version has no such fields — only override when present.
            if hasattr(ctrl, "T_robot_to_camera"):
                ctrl.cam_forward_offset = CAM_FORWARD_OFFSET
                ctrl.T_camera_to_control = ctrl.T_robot_to_camera.copy()
                ctrl.T_camera_to_control[2, 3] = -CAM_FORWARD_OFFSET
            # Sweep B's gains from the env without editing the deployment file.
            if "PNC_YAW_BIAS_KI" in os.environ and hasattr(ctrl, "yaw_bias_ki"):
                ctrl.yaw_bias_ki = float(os.environ["PNC_YAW_BIAS_KI"])
            if "PNC_DRIFT_TAU" in os.environ and hasattr(ctrl, "drift_filter_tau"):
                ctrl.drift_filter_tau = float(os.environ["PNC_DRIFT_TAU"])
            ctrl.cmd_timer.cancel()   # we tick it manually in sim time
            if fi == 0:
                print(f"robot: {cfg.name} (cam_fwd_offset={CAM_FORWARD_OFFSET:.2f}, "
                      f"safety={cfg.safety_radius})", flush=True)
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            executor.add_node(driver)
            executor.add_node(ctrl)

            for _ in range(SPIN_TRIES):       # latch camera_info -> node.K
                driver.publish_camera_info()
                executor.spin_once(timeout_sec=SPIN_DT)
                if getattr(node, "K", None) is not None:
                    break
            if getattr(node, "K", None) is None:
                print("  WARNING: node never received camera_info", flush=True)

            fp = cfg.footprint_from_control()
            for si, scene in enumerate(routes):
                res = run_route(node, ctrl, driver, executor, scene, fp)
                results[si][fi] = res
                traj = res["traj"]
                tag = f" {len(res['collisions'])}hits" if res["collisions"] else ""
                mdev, fdev = lateral_deviation(traj, scene.global_path_pts)
                bias = getattr(ctrl, "_yaw_bias_est", float("nan"))
                print(f"  {scene.name}: {len(traj) - 1} steps, "
                      f"end=({traj[-1, 0]:.2f},{traj[-1, 1]:.2f}) "
                      f"maxdev={mdev:.3f}m enddev={fdev:.3f}m learned_bias={bias:+.4f}{tag}",
                      flush=True)

            executor.remove_node(node)
            executor.remove_node(ctrl)
            node.destroy_node()
            ctrl.destroy_node()
    finally:
        rclpy.shutdown()

    for si, scene in enumerate(routes):
        out_path = os.path.join(args.out, f"{scene.name}.png")
        plot_route(scene, results[si], labels, out_path)
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
