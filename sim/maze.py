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
    NAMES = (
        "Teaching maze",
        "Classic labyrinth",
        "Room grid",
        "Switchbacks",
        "Pillar arena",
        "Random room",
    )

    def __init__(
        self,
        grid: np.ndarray,
        zones: Dict[str, ZoneBox],
        start: Agent,
        name: str = "Custom maze",
        seed: int | None = None,
    ):
        if grid.ndim != 2:
            raise ValueError("grid must be 2D")
        self.grid = grid.astype(np.int8, copy=True)
        self.zones = dict(zones)
        self.height, self.width = self.grid.shape
        self.start = start
        self.name = name
        self.seed = seed

    @classmethod
    def named(cls, name: str, seed: int | None = None) -> "Maze":
        """Build one of the built-in maze layouts by its display name."""
        builders = {
            "Teaching maze": cls.teaching_maze,
            "Classic labyrinth": cls.classic_labyrinth,
            "Room grid": cls.room_grid,
            "Switchbacks": cls.switchbacks,
            "Pillar arena": cls.pillar_arena,
        }
        if name == "Random room":
            actual_seed = (
                int(np.random.SeedSequence().generate_state(1)[0])
                if seed is None
                else seed
            )
            return cls.random_room(actual_seed)
        try:
            return builders[name]()
        except KeyError as exc:
            raise ValueError(f"Unknown maze layout: {name}") from exc

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
        maze = cls(grid, zones, start, "Teaching maze")
        if not maze.is_free(start.x, start.y):
            raise RuntimeError("teaching maze start pose is inside a wall")
        return maze

    @classmethod
    def classic_labyrinth(cls) -> "Maze":
        """Create a deterministic, connected depth-first-search labyrinth."""
        grid = np.ones((MAZE_SIZE, MAZE_SIZE), dtype=np.int8)
        rng = np.random.default_rng(17)
        stack = [(1, 1)]
        visited = {(1, 1)}
        grid[1, 1] = 0
        while stack:
            col, row = stack[-1]
            neighbors = [
                (col + dc, row + dr)
                for dc, dr in ((0, -2), (2, 0), (0, 2), (-2, 0))
                if 1 <= col + dc < MAZE_SIZE - 1
                and 1 <= row + dr < MAZE_SIZE - 1
                and (col + dc, row + dr) not in visited
            ]
            if not neighbors:
                stack.pop()
                continue
            next_col, next_row = neighbors[int(rng.integers(len(neighbors)))]
            grid[(row + next_row) // 2, (col + next_col) // 2] = 0
            grid[next_row, next_col] = 0
            visited.add((next_col, next_row))
            stack.append((next_col, next_row))
        return cls(grid, {}, Agent(1.5, 1.5, 0.0), "Classic labyrinth")

    @classmethod
    def room_grid(cls) -> "Maze":
        """Create nine open rooms connected through alternating doorways."""
        grid = np.ones((MAZE_SIZE, MAZE_SIZE), dtype=np.int8)
        grid[1:-1, 1:-1] = 0
        for wall in (13, 27):
            grid[1:-1, wall] = 1
            grid[1:5, wall] = 0
            grid[18:23, wall] = 0
            grid[MAZE_SIZE - 6 : MAZE_SIZE - 2, wall] = 0
            grid[wall, 1:-1] = 1
            grid[wall, 4:9] = 0
            grid[wall, 18:23] = 0
            grid[wall, MAZE_SIZE - 9 : MAZE_SIZE - 4] = 0
        return cls(grid, {}, Agent(6.0, 6.0, 0.0), "Room grid")

    @classmethod
    def switchbacks(cls) -> "Maze":
        """Create a broad serpentine route with alternating end openings."""
        grid = np.ones((MAZE_SIZE, MAZE_SIZE), dtype=np.int8)
        rows = list(range(2, MAZE_SIZE - 2, 6))
        for row in rows:
            grid[row : row + 3, 1:-1] = 0
        for index, row in enumerate(rows[:-1]):
            connector_col = MAZE_SIZE - 4 if index % 2 == 0 else 2
            grid[row : rows[index + 1] + 3, connector_col : connector_col + 3] = 0
        return cls(grid, {}, Agent(2.5, 3.0, 0.0), "Switchbacks")

    @classmethod
    def pillar_arena(cls) -> "Maze":
        """Create a spacious arena with offset block obstacles and narrow gates."""
        grid = np.ones((MAZE_SIZE, MAZE_SIZE), dtype=np.int8)
        grid[1:-1, 1:-1] = 0
        for row, col in (
            (5, 6), (5, 18), (5, 30),
            (15, 12), (15, 25),
            (25, 6), (25, 18), (25, 30),
            (34, 12), (34, 25),
        ):
            grid[row : row + 3, col : col + 3] = 1
        return cls(grid, {}, Agent(2.5, 2.5, 0.0), "Pillar arena")

    @classmethod
    def random_room(cls, seed: int) -> "Maze":
        """Generate a connected random room layout with seeded doors and partitions."""
        rng = np.random.default_rng(seed)

        def partitions() -> list[int]:
            count = int(rng.integers(2, 5))
            max_base = MAZE_SIZE - 7 - 5 * (count - 1)
            bases = np.sort(
                rng.choice(np.arange(7, max_base + 1), size=count, replace=False)
            )
            return [
                int(base + 5 * index) for index, base in enumerate(bases)
            ]

        vertical_walls = partitions()
        horizontal_walls = partitions()
        grid = np.ones((MAZE_SIZE, MAZE_SIZE), dtype=np.int8)
        grid[1:-1, 1:-1] = 0

        for col in vertical_walls:
            grid[1:-1, col] = 1
        for row in horizontal_walls:
            grid[row, 1:-1] = 1

        row_boundaries = [0, *horizontal_walls, MAZE_SIZE - 1]
        for col in vertical_walls:
            for row_start, row_end in zip(row_boundaries, row_boundaries[1:]):
                room_start, room_end = row_start + 1, row_end
                width = int(rng.integers(1, 3))
                door_start = int(rng.integers(room_start, room_end - width + 1))
                grid[door_start : door_start + width, col] = 0

        col_boundaries = [0, *vertical_walls, MAZE_SIZE - 1]
        for row in horizontal_walls:
            for col_start, col_end in zip(col_boundaries, col_boundaries[1:]):
                room_start, room_end = col_start + 1, col_end
                width = int(rng.integers(1, 3))
                door_start = int(rng.integers(room_start, room_end - width + 1))
                grid[row, door_start : door_start + width] = 0

        free_cells = np.argwhere(grid == 0)
        start_row, start_col = free_cells[int(rng.integers(len(free_cells)))]
        start = Agent(
            float(start_col) + 0.5,
            float(start_row) + 0.5,
            float(rng.uniform(-np.pi, np.pi)),
        )
        name = f"Random room (seed {seed})"
        return cls(grid, {}, start, name=name, seed=seed)

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
