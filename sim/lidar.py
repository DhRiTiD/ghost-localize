"""Simulated 2D LiDAR via ray marching on the occupancy grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .maze import Agent, Maze


@dataclass(frozen=True)
class LidarConfig:
    n_rays: int = 180  # doubled vs original 90
    max_range: float = 20.0  # covers the larger 41×41 map
    fov: float = 2.0 * np.pi  # full 360°
    step: float = 0.05


@dataclass
class LidarScan:
    angles: np.ndarray  # absolute world angles, radians
    ranges: np.ndarray  # meters / cell units
    hit_x: np.ndarray
    hit_y: np.ndarray

    @property
    def range_image(self) -> np.ndarray:
        """1 x N image in [0, 1], normalized by max finite range in this scan."""
        r = self.ranges.astype(np.float64)
        peak = float(np.max(r)) if r.size else 1.0
        if peak <= 0:
            peak = 1.0
        return (r / peak).reshape(1, -1)


def cast_lidar(maze: Maze, agent: Agent, config: LidarConfig | None = None) -> LidarScan:
    cfg = config or LidarConfig()
    rel = np.linspace(-cfg.fov / 2.0, cfg.fov / 2.0, cfg.n_rays, endpoint=False)
    angles = agent.theta + rel

    ranges = np.empty(cfg.n_rays, dtype=np.float64)
    hit_x = np.empty(cfg.n_rays, dtype=np.float64)
    hit_y = np.empty(cfg.n_rays, dtype=np.float64)

    for i, ang in enumerate(angles):
        c, s = float(np.cos(ang)), float(np.sin(ang))
        dist = 0.0
        x, y = agent.x, agent.y
        hit = False
        while dist < cfg.max_range:
            dist += cfg.step
            x = agent.x + c * dist
            y = agent.y + s * dist
            if not maze.is_free(x, y):
                ranges[i] = dist
                hit_x[i] = x
                hit_y[i] = y
                hit = True
                break
        if not hit:
            ranges[i] = cfg.max_range
            hit_x[i] = agent.x + c * cfg.max_range
            hit_y[i] = agent.y + s * cfg.max_range

    return LidarScan(angles=angles, ranges=ranges, hit_x=hit_x, hit_y=hit_y)
