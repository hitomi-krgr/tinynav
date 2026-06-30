"""Connectivity + small-elevated traversability building blocks.

Integrated via compute_walkable_obstacle() (called from planning_node's obstacle
map step). The key capabilities over main's z-span:
  - flood_walkable: connectivity from the robot foot -> reachable floor (stairs/
    ramps connect via small steps; cliffs / NaN block).
  - small_elevated_obstacles: white top-hat (local floor) -> low obstacles z-span
    misses (a lying robot's back, a low box); slopes/large platforms survive.
  - WalkableConfidence: temporal confidence to stabilize the per-frame decision.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, grey_opening, binary_fill_holes


@dataclass
class WalkableConfig:
    step_up_max: float = 0.22        # generous: bridge stairs despite 0.1m quantization
    step_down_max: float = 0.22
    connectivity: int = 8            # 4 or 8
    occ_threshold: float = 0.1       # same notion of "occupied" as z-span
    free_threshold: float = -0.02    # occ < this == raycast-through (known empty)
    height_band_m: float = 0.5       # +/- band around robot_z (grid is ~1m tall)
    rise_eps: float = 0.05           # "elevated" if > local floor + rise_eps
    min_walkable_extent_m2: float = 0.6   # (extent_filter) reachable raised blob
                                          # smaller than this -> obstacle
    obstacle_max_m: float = 0.8           # (small_elevated) top-hat window: raised
                                          # things smaller than this footprint are
                                          # obstacles; larger survive -> walkable
    union_zspan: bool = True


def extract_height_map(occ, origin, resolution, robot_z, cfg, z_band=None):
    """Per-(x,y) ground height = LOWEST occupied voxel within the z band relative
    to robot_z. Columns with no occupied voxel in band -> NaN (unknown). No
    smoothing. Returns (height, observed): observed = column had ANY occupancy
    evidence in band (occupied OR raycast-through free), i.e. seen this frame.

    z_band: optional (z_lo_rel, z_hi_rel) explicit band relative to robot_z. When
    None, falls back to the symmetric +/- cfg.height_band_m. Pass the planning
    grid's actual [robot_z_bottom, robot_z_top] so a high-mounted camera (floor
    well below robot_z) is still covered."""
    nz = occ.shape[2]
    zc = origin[2] + (np.arange(nz) + 0.5) * resolution
    z_rel = zc - robot_z
    if z_band is None:
        band = np.abs(z_rel) <= cfg.height_band_m
    else:
        band = (z_rel >= z_band[0]) & (z_rel <= z_band[1])
    if not band.any():
        shp = occ.shape[:2]
        return np.full(shp, np.nan, dtype=np.float32), np.zeros(shp, dtype=bool)
    occ_band = occ[:, :, band]
    occ_b = occ_band > cfg.occ_threshold
    has = occ_b.any(axis=2)
    zc_b = zc[band].astype(np.float32)
    first = np.argmax(occ_b, axis=2)            # first True == lowest occupied
    h = np.where(has, zc_b[first], np.nan).astype(np.float32)
    observed = has | (occ_band < cfg.free_threshold).any(axis=2)
    return h, observed


def small_elevated_obstacles(height, floor_h, resolution, cfg):
    """Low obstacles z-span misses (thin-topped: a lying robot's back, a box):
    cells that stick up above their LOCAL surroundings (white top-hat). Uses a
    LOCAL floor (morphological opening) so a sloped/uneven floor is NOT flagged --
    only protrusions smaller than obstacle_max_m are. Larger raised things
    (stairs/ramp/wide platform) survive the opening -> not flagged. Unknown (NaN)
    -> not elevated. Returns a bool obstacle mask."""
    W = max(3, int(round(cfg.obstacle_max_m / resolution)))
    filled = np.where(np.isfinite(height), height, floor_h).astype(np.float32)
    local_floor = grey_opening(filled, size=(W, W))     # floor under small bumps
    return np.isfinite(height) & (height - local_floor > cfg.rise_eps)


def flood_walkable(height, seed_cells, seed_h, cfg):
    """BFS from seeds across finite-height cells within climb limits (up/down).
    NaN (unknown) cells block. Local-adjacency step test lets stairs/ramps connect.
    Returns bool walkable mask."""
    nx, ny = height.shape
    walk = np.zeros((nx, ny), dtype=bool)
    work = np.where(np.isnan(height), 0.0, height).astype(np.float32)
    q = deque()
    for (i, j) in seed_cells:
        if 0 <= i < nx and 0 <= j < ny and not walk[i, j]:
            walk[i, j] = True
            work[i, j] = height[i, j] if np.isfinite(height[i, j]) else seed_h
            q.append((i, j))
    if cfg.connectivity == 8:
        nbrs = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    else:
        nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while q:
        i, j = q.popleft()
        hc = work[i, j]
        for di, dj in nbrs:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or walk[ni, nj]:
                continue
            hn = height[ni, nj]
            if hn != hn:                         # NaN blocks
                continue
            dz = hn - hc
            if dz <= cfg.step_up_max and -dz <= cfg.step_down_max:
                walk[ni, nj] = True
                work[ni, nj] = hn
                q.append((ni, nj))
    return walk


def extent_filter(walk, height, floor_h, resolution, cfg):
    """Remove small isolated ELEVATED walkable blobs (dog/box) from the reachable
    mask -> obstacle. Large elevated regions (stairs/ramp, spatially contiguous)
    are kept walkable. Returns (walk_filtered, removed_mask)."""
    out = walk.copy()
    elevated = walk & np.isfinite(height) & (height > floor_h + cfg.rise_eps)
    removed = np.zeros_like(walk)
    if elevated.any():
        lbl, n = label(elevated, structure=np.ones((3, 3)))
        min_cells = cfg.min_walkable_extent_m2 / (resolution * resolution)
        for c in range(1, n + 1):
            comp = lbl == c
            if comp.sum() < min_cells:
                removed |= comp
        out &= ~removed
    return out, removed


class WalkableConfidence:
    """Temporal walkability confidence (occupancy-grid style): per-cell value in
    [-clip, clip]; each frame decay, +hit where walkable, +miss on AFFIRMATIVE
    obstacle evidence (NOT mere flood-miss). Hysteresis output + bad-frame guard.
    Rolls with the robot like the occupancy grid."""

    def __init__(self, shape_xy, hit=0.2, miss=-0.05, decay=0.995,
                 clip=0.2, on_thresh=0.06, off_thresh=-0.06, bad_frame_ratio=0.7):
        self.conf = np.zeros(shape_xy, dtype=np.float32)
        self.on = np.zeros(shape_xy, dtype=bool)
        self.hit, self.miss, self.decay = hit, miss, decay
        self.clip = clip
        self.on_thresh, self.off_thresh = on_thresh, off_thresh
        self.bad_frame_ratio = bad_frame_ratio
        self._ema_walk = None
        self._origin_xy = None
        self.last_bad = False
        self.last_roll_cells = 0

    def _roll_to(self, origin_xy, resolution):
        if self._origin_xy is None:
            self._origin_xy = np.asarray(origin_xy, dtype=np.float64)
            return
        shift = np.round((np.asarray(origin_xy) - self._origin_xy) / resolution).astype(int)
        self.last_roll_cells = int(abs(shift[0]) + abs(shift[1]))
        if shift[0] or shift[1]:
            roll = (-shift[0], -shift[1])
            self.conf = np.roll(self.conf, shift=roll, axis=(0, 1))
            self.on = np.roll(self.on, shift=roll, axis=(0, 1))
            for arr, fill in ((self.conf, 0.0), (self.on, False)):
                if shift[0] > 0:
                    arr[-shift[0]:, :] = fill
                elif shift[0] < 0:
                    arr[:-shift[0], :] = fill
                if shift[1] > 0:
                    arr[:, -shift[1]:] = fill
                elif shift[1] < 0:
                    arr[:, :-shift[1]] = fill
            self._origin_xy = self._origin_xy + shift * resolution

    def update(self, walkable_now, not_walkable, origin_xy, resolution):
        """hit on cells reachable this frame; miss ONLY on affirmative obstacle
        evidence (so floor cut off by a transient gap keeps its confidence).
        Bad-frame guard: skip miss when reachable area collapses far below the
        recent peak. Returns the stabilized walkable bool mask."""
        self._roll_to(origin_xy, resolution)
        n_walk = int(walkable_now.sum())
        if self._ema_walk is None:
            self._ema_walk = float(n_walk)
        bad = self._ema_walk > 100 and n_walk < self.bad_frame_ratio * self._ema_walk
        self.last_bad = bool(bad)
        self.conf *= self.decay
        self.conf[walkable_now] += self.hit
        if not bad:
            self.conf[not_walkable] += self.miss
        np.clip(self.conf, -self.clip, self.clip, out=self.conf)
        self.on = np.where(self.conf > self.on_thresh, True,
                           np.where(self.conf < self.off_thresh, False, self.on))
        e = 0.3 if n_walk > self._ema_walk else 0.02   # ema tracks recent peak
        self._ema_walk = (1 - e) * self._ema_walk + e * n_walk
        return self.on


def estimate_floor_height(height, center_cell, radius_cells, robot_z,
                          fallback_drop=0.4):
    """Floor height under/around the robot = low percentile of finite heights in a
    window around the robot center (the footprint is often NaN -> look at the
    neighborhood). Fallback: robot_z - fallback_drop (~ robot_z + robot_z_bottom)."""
    ci, cj = center_cell
    nx, ny = height.shape
    i0, i1 = max(0, ci - radius_cells), min(nx, ci + radius_cells + 1)
    j0, j1 = max(0, cj - radius_cells), min(ny, cj + radius_cells + 1)
    win = height[i0:i1, j0:j1]
    finite = win[np.isfinite(win)]
    if finite.size >= 5:
        return float(np.percentile(finite, 25))
    return float(robot_z - fallback_drop)


def compute_walkable_obstacle(occupancy_grid, origin, resolution, robot_z, zspan,
                              cfg=None, conf=None, fallback_drop=0.4, z_band=None):
    """Connectivity-based obstacle mask, integrating the z-span result.

    Approach: floor reachable from the robot foot via small steps is walkable
    (stairs/ramps connect, walls/cliffs block). Low obstacles the z-span misses
    are added back via a local top-hat. An optional WalkableConfidence (conf)
    stabilizes the per-frame decision across frames.

    Args:
        zspan: precomputed z-span obstacle mask (build_obstacle_map result),
            passed in to avoid a circular import on planning_node.
        cfg: WalkableConfig (defaults if None).
        conf: optional WalkableConfidence for temporal smoothing; pass the same
            instance every frame. None => per-frame only.
        fallback_drop: floor-height fallback below robot_z when the footprint
            neighborhood has no finite height (use -robot_z_bottom).
        z_band: optional (z_lo_rel, z_hi_rel) height band relative to robot_z;
            pass the grid's [robot_z_bottom, robot_z_top] so a high-mounted
            camera still sees the floor. None => symmetric +/- cfg.height_band_m.

    Returns:
        (NX, NY) bool obstacle mask (obstacle = ~walkable).
    """
    cfg = cfg or WalkableConfig()
    nx, ny = occupancy_grid.shape[:2]

    height, observed = extract_height_map(occupancy_grid, origin, resolution, robot_z, cfg, z_band=z_band)

    ci, cj = nx // 2, ny // 2
    seeds = [(i, j) for i in range(ci - 2, ci + 3) for j in range(cj - 2, cj + 3)
             if 0 <= i < nx and 0 <= j < ny]
    floor_h = estimate_floor_height(height, (ci, cj), radius_cells=10,
                                    robot_z=robot_z, fallback_drop=fallback_drop)

    walk = flood_walkable(height, seeds, floor_h, cfg)
    walk, removed = extent_filter(walk, height, floor_h, resolution, cfg)
    walkable_now = walk & ~zspan

    if conf is not None:
        not_walkable = (zspan | removed) & observed
        stable = conf.update(walkable_now, not_walkable, origin[:2], resolution)
    else:
        stable = walkable_now

    affirmative_obs = zspan | small_elevated_obstacles(height, floor_h, resolution, cfg)
    solid = binary_fill_holes(stable)                 # enclosed unknown -> walkable
    walkable_final = solid & ~affirmative_obs
    for (i, j) in seeds:
        walkable_final[i, j] = True                   # never trap on own footprint

    return ~walkable_final
