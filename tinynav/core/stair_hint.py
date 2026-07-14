"""Stair hint from the capture trajectory.

Rationale: with 先建图后导航 the global planner stays near the capture path
(SDF seeds = capture trajectory). So "am I heading into stairs?" reduces to
"does the capture path here go up/down a sustained flight?". We label each
capture-path sample offline (at build time) as climbing/flat using a SUSTAINED
NET vertical change over a horizontal window — robust to quadruped gait bob and
to cresting a small ramp (both are transient, near-zero net change), and to VIO
teleports (rejected by a sign-consistency test). At nav time a tiny node looks
up the nearest labelled sample to the robot's pose-in-map and emits a boolean.

Pure numpy; no ROS, no occupancy grid — rides on `poses.npy` which build_map
already saves.
"""
from __future__ import annotations
import numpy as np

# Defaults (tunable). See discussion: 0.25 m net rise over a +/-1 m window trades
# single-step sensitivity for gait robustness (a ~5 cm bob nets ~0).
WIN_M = 1.0            # half-window horizontal arclength (m)
MIN_RISE = 0.25        # min sustained net |dz| over the window to call it climbing (m)
CONSISTENCY = 0.6      # min fraction of in-window steps whose dz sign matches the net
MAX_STEP_DZ = 0.5      # a single consecutive |dz| above this = VIO teleport -> not climbing
LOOKAHEAD_M = 1.5      # nav-time lead: on_stairs fires if a climbing sample is within this
                       # 3D radius of the robot -> opens z-span ~this far before the flight.
                       # Note the +/-WIN_M labelling window already spreads climbing labels
                       # ~WIN_M ahead of the first riser, so effective lead ~= WIN_M + this.


def poses_to_positions(poses) -> np.ndarray:
    """poses: dict{timestamp:int -> 4x4} (as saved by build_map). Returns (N,3)
    translations ordered by timestamp (== capture order)."""
    if isinstance(poses, dict):
        keys = sorted(poses.keys())
        return np.array([np.asarray(poses[k])[:3, 3] for k in keys], dtype=np.float64)
    arr = np.asarray(poses, dtype=np.float64)
    return arr[:, :3, 3] if arr.ndim == 3 else arr[:, :3]


def compute_path_climb(poses, win_m=WIN_M, min_rise=MIN_RISE,
                       consistency=CONSISTENCY, max_step_dz=MAX_STEP_DZ) -> np.ndarray:
    """Label each capture-path sample climbing (1) or flat (0).

    Returns (N,4) float array: columns [x, y, z, is_climbing]. is_climbing is
    1.0 where the path shows a sustained net vertical change over a +/-win_m
    horizontal window with consistent sign (robust to gait bob / ramp crest).
    """
    pos = poses_to_positions(poses)
    n = len(pos)
    out = np.zeros((n, 4), dtype=np.float32)
    if n == 0:
        return out
    out[:, :3] = pos
    if n < 3:
        return out
    # cumulative horizontal arclength
    dxy = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(dxy)])
    z = pos[:, 2]
    for i in range(n):
        j0 = np.searchsorted(s, s[i] - win_m, side='left')
        j1 = np.searchsorted(s, s[i] + win_m, side='right')
        if j1 - j0 < 3:
            continue
        seg_z = z[j0:j1]
        steps = np.diff(seg_z)
        if steps.size == 0 or np.abs(steps).max() > max_step_dz:
            continue                       # teleport / discontinuity in window
        net = seg_z[-1] - seg_z[0]
        if abs(net) < min_rise:
            continue
        same = np.mean(np.sign(steps) == np.sign(net))
        if same >= consistency:
            out[i, 3] = 1.0
    return out


class PathClimbIndex:
    """Nav-time lookup: is a climbing-labelled capture-path sample near the robot?"""

    def __init__(self, path_climb: np.ndarray, lookahead_m: float = LOOKAHEAD_M):
        self.pts = np.asarray(path_climb, dtype=np.float64)
        self.lookahead_m = float(lookahead_m)
        climb = self.pts[:, 3] >= 0.5 if self.pts.shape[0] else np.zeros(0, bool)
        self._climb_xyz = self.pts[climb, :3]      # (M,3) climbing samples only

    @classmethod
    def load(cls, npy_path: str, **kw) -> "PathClimbIndex":
        return cls(np.load(npy_path), **kw)

    def on_stairs(self, position_xyz) -> bool:
        """True if any climbing-labelled capture-path sample is within lookahead_m
        (full 3D distance, so stacked floors don't alias) of the robot -> gives a
        ~lookahead_m lead before the flight. Off-path / no data -> False (unknown
        => strict z-span, the safe default)."""
        if self._climb_xyz.shape[0] == 0:
            return False
        p = np.asarray(position_xyz, dtype=np.float64)[:3]
        d3 = np.linalg.norm(self._climb_xyz - p, axis=1)
        return bool(d3.min() <= self.lookahead_m)
