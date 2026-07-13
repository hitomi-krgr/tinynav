"""Stair / step carve-out on top of a conservative z-span obstacle mask.

Design contract (agreed direction):
  - The z-span obstacle mask is run at a LOW span threshold so it conservatively
    marks EVERY low protrusion as obstacle -- real low obstacles, stair risers,
    single steps alike. That mask is the safety floor: default is "block".
  - This module only does SUBTRACTION: it identifies cells that are part of a
    genuine stair / step / ramp structure and carves them OUT of the obstacle
    mask (marks them passable), so the robot can drive onto them and the
    low-level locomotion controller handles the actual climbing gait.
  - Everything not positively confirmed as a step stays an obstacle: low boxes,
    reflection phantoms, walls. Failure direction is therefore SAFE (a missed
    step means "can't climb this time", never "drive into an obstacle").

Discriminator (what tells a step from a same-height box):
  A step/stair is a RISE followed by a SUSTAINED elevated walkable surface that
  (a) does not return to the base floor over a forward window, (b) rises in
  climb-sized increments (<= climb_step_max), and (c) spans a lateral extent
  across the travel corridor. An isolated bump (box / reflection) rises then
  the surface returns to floor, or has no lateral extent -> NOT carved.

Reflection robustness: specular floor phantoms are isolated, non-monotonic, and
lack sustained forward + lateral extent, so they fail the template and stay
blocked. No global flood / NaN-hole reasoning (that is what broke the earlier
connectivity 'walkable' layer under reflections).
"""
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.ndimage import label


@dataclass
class StairConfig:
    occ_threshold: float = 0.1        # same "occupied" notion as z-span
    z_lo_rel: float = -0.4            # z band relative to robot_z (grid extent)
    z_hi_rel: float = 0.4
    climb_step_max: float = 0.20      # max single-step rise the robot can climb;
                                      # rises larger than this per step -> wall, not carved
    min_rise: float = 0.04            # "elevated" if surface > base_floor + min_rise
    forward_window_m: float = 0.6     # distance ahead the elevation must be SUSTAINED
    return_tol: float = 0.06          # surface within this of base_floor == "returned"
    lateral_extent_m: float = 0.4     # elevated structure must span >= this laterally
    min_component_m2: float = 0.15    # drop tiny carved blobs (box-top sized)


def surface_height_map(occ, origin, resolution, robot_z, cfg):
    """Per-(x,y) walkable surface height = TOP occupied voxel within the z band,
    relative to robot_z. Columns with no occupancy in band -> NaN (unknown).
    Top (not bottom) surface: that is the tread/box-top you would stand on, and
    it is more stable than the lowest voxel under floor reflections."""
    nz = occ.shape[2]
    zc = origin[2] + (np.arange(nz) + 0.5) * resolution
    z_rel = zc - robot_z
    band = (z_rel >= cfg.z_lo_rel) & (z_rel <= cfg.z_hi_rel)
    if not band.any():
        shp = occ.shape[:2]
        return np.full(shp, np.nan, dtype=np.float32)
    occ_b = occ[:, :, band] > cfg.occ_threshold
    has = occ_b.any(axis=2)
    zc_b = z_rel[band].astype(np.float32)
    top = occ_b.shape[2] - 1 - np.argmax(occ_b[:, :, ::-1], axis=2)  # highest True
    h = np.where(has, zc_b[top], np.nan).astype(np.float32)
    return h


def estimate_base_floor(height, center_cell, radius_cells, robot_z, fallback_drop):
    """Base floor the robot stands on = low percentile of finite surface heights in
    a window around the robot center. Fallback: robot_z - fallback_drop."""
    ci, cj = center_cell
    nx, ny = height.shape
    i0, i1 = max(0, ci - radius_cells), min(nx, ci + radius_cells + 1)
    j0, j1 = max(0, cj - radius_cells), min(ny, cj + radius_cells + 1)
    win = height[i0:i1, j0:j1]
    finite = win[np.isfinite(win)]
    if finite.size >= 5:
        return float(np.percentile(finite, 25))
    return float(-fallback_drop)


@njit(cache=True)
def _carve_kernel(height, elevated, candidate, floor, di_f, dj_f, latv0, latv1,
                  n_fwd, n_lat, return_tol, climb_step_max):
    """Per-candidate forward-profile + lateral-extent test (hot loop).

    For each elevated obstacle candidate cell: carve it iff the surface ahead
    stays elevated (never returns to floor within the forward window) rising in
    climb-sized steps, AND the elevation spans the lateral corridor band."""
    nx, ny = height.shape
    carve = np.zeros((nx, ny), dtype=np.bool_)
    for i in range(nx):
        for j in range(ny):
            if not candidate[i, j]:
                continue
            # (a) SUSTAINED forward: not returning to floor, climb-sized rises.
            sustained = True
            prev = height[i, j]
            for s in range(1, n_fwd + 1):
                ai = int(round(i + di_f * s))
                aj = int(round(j + dj_f * s))
                if ai < 0 or ai >= nx or aj < 0 or aj >= ny:
                    sustained = False
                    break
                ha = height[ai, aj]
                if ha != ha:                    # NaN: unknown ahead, skip
                    continue
                if ha <= floor + return_tol:    # returned to floor -> bump, not step
                    sustained = False
                    break
                if ha - prev > climb_step_max:  # too-tall single jump -> wall
                    sustained = False
                    break
                if ha > prev:
                    prev = ha
            if not sustained:
                continue
            # (b) LATERAL EXTENT across the corridor.
            lat_hits = 0
            for t in range(-n_lat, n_lat + 1):
                li = int(round(i + latv0 * t))
                lj = int(round(j + latv1 * t))
                if 0 <= li < nx and 0 <= lj < ny and elevated[li, lj]:
                    lat_hits += 1
            if lat_hits < n_lat:
                continue
            carve[i, j] = True
    return carve


def detect_stair_carveout(occupancy_grid, origin, resolution, robot_z,
                          forward_xy, center_cell, zspan_mask, cfg=None,
                          fallback_drop=0.4):
    """Return a bool carve mask (True = this obstacle cell is a confirmed step and
    should be REMOVED from the obstacle mask).

    Args:
        forward_xy: robot forward direction in world XY (unit-ish 2-vector).
        center_cell: (i, j) grid cell of the robot control center.
        zspan_mask: the conservative (low-threshold) z-span obstacle mask.
    Only cells set in zspan_mask can ever be carved.
    """
    cfg = cfg or StairConfig()
    nx, ny = occupancy_grid.shape[:2]
    height = surface_height_map(occupancy_grid, origin, resolution, robot_z, cfg)
    floor = estimate_base_floor(height, center_cell, radius_cells=10,
                                robot_z=robot_z, fallback_drop=fallback_drop)

    f = np.asarray(forward_xy, dtype=np.float64)
    n = np.linalg.norm(f)
    if n < 1e-6:
        return np.zeros((nx, ny), dtype=bool)
    f = f / n                                   # forward unit (world XY)
    latv = np.array([-f[1], f[0]])              # left, perpendicular
    di_f, dj_f = f                              # cells indexed [x, y] == world XY / res
    n_fwd = max(1, int(round(cfg.forward_window_m / resolution)))
    n_lat = max(1, int(round(cfg.lateral_extent_m / resolution)))

    elevated = np.isfinite(height) & (height > floor + cfg.min_rise)
    candidate = zspan_mask & elevated           # only elevated obstacle cells

    carve = _carve_kernel(
        height, elevated, candidate, float(floor),
        float(di_f), float(dj_f), float(latv[0]), float(latv[1]),
        int(n_fwd), int(n_lat),
        float(cfg.return_tol), float(cfg.climb_step_max),
    )

    # (c) drop tiny carved components (box-top sized survivors)
    if carve.any():
        lbl, ncomp = label(carve, structure=np.ones((3, 3)))
        min_cells = cfg.min_component_m2 / (resolution * resolution)
        for c in range(1, ncomp + 1):
            comp = lbl == c
            if comp.sum() < min_cells:
                carve[comp] = False
    return carve
