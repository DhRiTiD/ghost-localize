from __future__ import annotations

from collections import deque
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim.dataset import generate_dataset, inspect_dataset, save_dataset
from sim.lidar import LidarConfig, cast_lidar
from sim.maze import Agent, Maze
from sim.render import FREE, OCCUPIED, blank_explored, integrate_scan


class LidarContractTests(unittest.TestCase):
    def test_full_scan_has_ordered_angles_and_capped_ranges(self) -> None:
        maze = Maze.teaching_maze()
        config = LidarConfig(n_rays=36, max_range=3.0, step=0.05)
        scan = cast_lidar(maze, maze.start, config)

        self.assertEqual(scan.ranges.shape, (36,))
        self.assertEqual(scan.hit_mask.shape, (36,))
        self.assertAlmostEqual(scan.angles[0], maze.start.theta - np.pi)
        self.assertLessEqual(float(scan.ranges.max()), config.max_range)
        self.assertTrue(np.isfinite(scan.ranges).all())

    def test_max_range_returns_are_not_marked_as_wall_hits(self) -> None:
        maze = Maze.pillar_arena()
        pose = Agent(20.5, 20.5, 0.0)
        scan = cast_lidar(maze, pose, LidarConfig(n_rays=4, max_range=0.1, step=0.05))
        self.assertFalse(scan.hit_mask.any())
        np.testing.assert_allclose(scan.ranges, 0.1)

    def test_scan_integration_only_marks_real_hits_occupied(self) -> None:
        maze = Maze.pillar_arena()
        pose = Agent(20.5, 20.5, 0.0)
        config = LidarConfig(n_rays=4, max_range=0.1, step=0.05)
        scan = cast_lidar(maze, pose, config)
        explored = integrate_scan(blank_explored(maze), maze, pose, scan)
        self.assertFalse(np.any(explored == OCCUPIED))
        self.assertEqual(explored[20, 20], FREE)


class DatasetTests(unittest.TestCase):
    def test_dataset_is_reproducible_and_split_by_maze(self) -> None:
        first = generate_dataset(samples_per_maze=2, seed=3, random_rooms_per_split=0)
        second = generate_dataset(samples_per_maze=2, seed=3, random_rooms_per_split=0)

        np.testing.assert_array_equal(first["scans"], second["scans"])
        np.testing.assert_array_equal(first["poses"], second["poses"])
        self.assertEqual(first["maps"].shape, (5, 41, 41))
        self.assertEqual(first["map_ids"].shape, (10,))
        self.assertEqual(first["scans"].shape, (10, 180))
        self.assertEqual(first["hit_mask"].shape, (10, 180))
        self.assertEqual(first["poses"].shape, (10, 4))
        metadata = json.loads(str(first["metadata_json"].item()))
        self.assertEqual(metadata["dataset_version"], 3)
        self.assertEqual(metadata["split_mazes"]["test"], ["Pillar arena"])

        for name, split, map_id, pose in zip(
            first["maze_names"],
            first["splits"],
            first["map_ids"],
            first["poses"],
        ):
            self.assertEqual(
                split,
                {
                    "Teaching maze": "train",
                    "Classic labyrinth": "train",
                    "Switchbacks": "train",
                    "Room grid": "validation",
                    "Pillar arena": "test",
                }[name],
            )
            maze = Maze.named(name)
            np.testing.assert_array_equal(first["maps"][map_id], maze.grid)
            self.assertTrue(maze.is_free(float(pose[0]), float(pose[1])))
            self.assertAlmostEqual(float(np.hypot(pose[2], pose[3])), 1.0, places=5)

    def test_dataset_round_trips_and_can_be_inspected(self) -> None:
        dataset = generate_dataset(
            samples_per_maze=1,
            seed=4,
            random_rooms_per_split=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_dataset(dataset, Path(directory) / "scans.npz")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["maps"], dataset["maps"])
                np.testing.assert_array_equal(saved["map_ids"], dataset["map_ids"])
                np.testing.assert_array_equal(saved["scans"], dataset["scans"])
                np.testing.assert_array_equal(saved["hit_mask"], dataset["hit_mask"])
            inspect_dataset(path)

    def test_random_rooms_are_reproducible_connected_and_split_by_room(self) -> None:
        first = generate_dataset(
            samples_per_maze=1,
            seed=17,
            random_rooms_per_split=2,
            samples_per_random_room=1,
        )
        second = generate_dataset(
            samples_per_maze=1,
            seed=17,
            random_rooms_per_split=2,
            samples_per_random_room=1,
        )
        metadata = json.loads(str(first["metadata_json"].item()))
        room_ids = metadata["random_room_ids"]

        np.testing.assert_array_equal(first["maze_names"], second["maze_names"])
        np.testing.assert_array_equal(first["scans"], second["scans"])
        self.assertEqual(len(set(room_ids["train"])), 2)
        self.assertEqual(len(set(room_ids["validation"])), 2)
        self.assertEqual(len(set(room_ids["test"])), 2)
        self.assertFalse(set(room_ids["train"]) & set(room_ids["validation"]))
        self.assertFalse(set(room_ids["train"]) & set(room_ids["test"]))
        self.assertFalse(set(room_ids["validation"]) & set(room_ids["test"]))
        for split_name, room_names in room_ids.items():
            for room_name in room_names:
                map_id = next(
                    record["map_id"]
                    for record in metadata["maps"]
                    if record["name"] == room_name
                )
                maze_seed = int(
                    room_name.removeprefix("Random room (seed ").removesuffix(")")
                )
                np.testing.assert_array_equal(
                    first["maps"][map_id],
                    Maze.random_room(maze_seed).grid,
                )
                self.assertTrue(
                    np.all(first["splits"][first["map_ids"] == map_id] == split_name)
                )

        for room_name in sum(room_ids.values(), []):
            seed = int(room_name.removeprefix("Random room (seed ").removesuffix(")"))
            maze = Maze.random_room(seed)
            start = maze.cell(maze.start.x, maze.start.y)
            queue = deque([start])
            visited = {start}
            while queue:
                col, row = queue.popleft()
                for neighbor in (
                    (col + 1, row),
                    (col - 1, row),
                    (col, row + 1),
                    (col, row - 1),
                ):
                    x, y = neighbor
                    if neighbor not in visited and maze.is_free(x + 0.5, y + 0.5):
                        visited.add(neighbor)
                        queue.append(neighbor)
            self.assertEqual(len(visited), int((maze.grid == 0).sum()), room_name)


if __name__ == "__main__":
    unittest.main()
