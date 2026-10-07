"""2D occupancy maze with deliberate teaching zones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# grid[row, col] == grid[y, x] ; 1 = wall, 0 = free
ZoneBox = Tuple[int, int, int, int]  # col0, row0, col1, row1 (inclusive)

# Reasonable larger teaching map (was 21×21)
MAZE_SIZE = 41


@dataclass
class Agent:
    x: float  # continuous cell units (column / +x)
    y: float  # continuous cell units (row / +y)
    theta: float  # radians; 0 = +x (right), pi/2 = +y (down in image coords)


class Maze:
    def __init__(self, grid: np.ndarray, zones: Dict[str, ZoneBox], start: Agent):
        if grid.ndim != 2:
            raise ValueError("grid must be 2D")
        self.grid = grid.astype(np.int8, copy=True)
        self.zones = dict(zones)
        self.height, self.width = self.grid.shape
        self.start = start

    @classmethod
    def teaching_maze(cls) -> "Maze":
        """
        Fixed 41x41 maze with three teaching zones:
          - symmetric: long parallel corridor (ambiguous scans)
          - sparse: large open room (few features)
          - rich: alcoves / corners (distinct structure)
        """
        h = w = MAZE_SIZE
        grid = np.ones((h, w), dtype=np.int8)

        # Carve interior free
        grid[1 : h - 1, 1 : w - 1] = 0

        # --- Symmetric corridor (left): free cols 5-7, walls at 4 and 8 ---
        grid[1 : h - 1, 4] = 1
        grid[1 : h - 1, 8] = 1
        grid[1 : h - 1, 5:8] = 0
        # openings into the rest of the map
        grid[1:4, 8] = 0
        grid[h - 4 : h - 1, 8] = 0

        # Divider wall between mid and sparse room, with a doorway
        divider = 17
        grid[1 : h - 1, divider] = 1
        grid[18:23, divider] = 0

        # --- Sparse open room (right) ---
        grid[3:33, 18:38] = 0

        # --- Rich feature zone (bottom-center): jutting blocks / alcoves ---
        rich_blocks = [
            (11, 27),
            (13, 31),
            (15, 29),
            (17, 33),
            (19, 27),
            (21, 31),
            (23, 29),
            (25, 33),
            (13, 35),
            (21, 35),
            (15, 25),
            (23, 25),
            (12, 30),
            (24, 30),
            (18, 28),
            (20, 34),
        ]
        for c, r in rich_blocks:
            if 0 <= r < h and 0 <= c < w:
                grid[r, c] = 1

        # Small pockets (alcoves) along the bottom corridor
        grid[h - 2, 11:16] = 0
        grid[h - 3, 11] = 1
        grid[h - 3, 13] = 1
        grid[h - 3, 15] = 1

        zones: Dict[str, ZoneBox] = {
            "symmetric": (4, 1, 8, 33),
            "sparse": (18, 3, 37, 32),
            "rich": (11, 25, 25, 38),
        }
        start = Agent(x=6.5, y=20.5, theta=np.pi / 2)  # face down the corridor
        maze = cls(grid, zones, start)
        if not maze.is_free(start.x, start.y):
            raise RuntimeError("teaching maze start pose is inside a wall")
        return maze

    def in_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x < self.width and 0.0 <= y < self.height

    def cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(np.floor(x)), int(np.floor(y))

    def is_free(self, x: float, y: float) -> bool:
        if not self.in_bounds(x, y):
            return False
        col, row = self.cell(x, y)
        return self.grid[row, col] == 0

    def zone_at(self, x: float, y: float) -> Optional[str]:
        col, row = self.cell(x, y)
        for name, (c0, r0, c1, r1) in self.zones.items():
            if c0 <= col <= c1 and r0 <= row <= r1:
                return name
        return None

    def try_move(self, agent: Agent, forward: float, dtheta: float) -> Agent:
        """Rotate first, then step. Reject translation if endpoint hits a wall."""
        theta = agent.theta + dtheta
        nx = agent.x + forward * float(np.cos(theta))
        ny = agent.y + forward * float(np.sin(theta))
        if forward == 0.0 or self.is_free(nx, ny):
            return Agent(x=nx if forward != 0.0 else agent.x, y=ny if forward != 0.0 else agent.y, theta=theta)
        # allow spin-in-place even when blocked ahead
        return Agent(x=agent.x, y=agent.y, theta=theta)
