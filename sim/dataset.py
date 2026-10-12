"""Generate and inspect reproducible LiDAR-to-pose localization datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .lidar import LidarConfig, cast_lidar
from .maze import Agent, Maze

DATASET_VERSION = 3
FIXED_MAZES = tuple(name for name in Maze.NAMES if name != "Random room")
TRAIN_MAZES = ("Teaching maze", "Classic labyrinth", "Switchbacks")
VALIDATION_MAZES = ("Room grid",)
TEST_MAZES = ("Pillar arena",)
SPLIT_BY_MAZE = {
    **{name: "train" for name in TRAIN_MAZES},
    **{name: "validation" for name in VALIDATION_MAZES},
    **{name: "test" for name in TEST_MAZES},
}
POSE_LABELS = ("x_cells", "y_cells", "heading_cos", "heading_sin")


def _sample_pose(maze: Maze, rng: np.random.Generator) -> tuple[float, float, float]:
    free_cells = np.argwhere(maze.grid == 0)
    for _ in range(100):
        row, col = free_cells[int(rng.integers(len(free_cells)))]
        x = float(col + rng.uniform(0.2, 0.8))
        y = float(row + rng.uniform(0.2, 0.8))
        if maze.is_free(x, y):
            return x, y, float(rng.uniform(-np.pi, np.pi))
    raise RuntimeError(f"Could not sample a valid pose in {maze.name}")


def generate_dataset(
    samples_per_maze: int = 500,
    seed: int = 42,
    config: LidarConfig | None = None,
    random_rooms_per_split: int = 5,
    samples_per_random_room: int = 100,
) -> dict[str, Any]:
    """Generate fixed- and random-maze scans with pose labels and room-held-out splits."""
    if samples_per_maze < 1:
        raise ValueError("samples_per_maze must be at least 1")
    if random_rooms_per_split < 0:
        raise ValueError("random_rooms_per_split cannot be negative")
    if random_rooms_per_split and samples_per_random_room < 1:
        raise ValueError("samples_per_random_room must be at least 1")
    unknown = set(FIXED_MAZES) - set(SPLIT_BY_MAZE)
    if unknown:
        raise RuntimeError(f"Mazes have no dataset split assignment: {sorted(unknown)}")
    cfg = config or LidarConfig()
    rng = np.random.default_rng(seed)
    random_room_ids: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    rooms: list[tuple[Maze, str, int]] = [
        (Maze.named(name), SPLIT_BY_MAZE[name], samples_per_maze)
        for name in FIXED_MAZES
    ]
    room_seeds: set[int] = set()
    for split in ("train", "validation", "test"):
        for _ in range(random_rooms_per_split):
            room_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
            while room_seed in room_seeds:
                room_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
            room_seeds.add(room_seed)
            maze = Maze.random_room(room_seed)
            random_room_ids[split].append(maze.name)
            rooms.append((maze, split, samples_per_random_room))

    count = sum(sample_count for _, _, sample_count in rooms)
    maps = np.stack([maze.grid.astype(np.uint8) for maze, _, _ in rooms])
    map_ids = np.empty(count, dtype=np.int32)
    scans = np.empty((count, cfg.n_rays), dtype=np.float32)
    hit_mask = np.empty((count, cfg.n_rays), dtype=np.bool_)
    poses = np.empty((count, len(POSE_LABELS)), dtype=np.float32)
    max_name_length = max(len(maze.name) for maze, _, _ in rooms)
    maze_names = np.empty(count, dtype=f"<U{max_name_length}")
    splits = np.empty(count, dtype="<U10")

    map_records = []
    row = 0
    for map_id, (maze, split, sample_count) in enumerate(rooms):
        map_records.append(
            {
                "map_id": map_id,
                "name": maze.name,
                "seed": maze.seed,
                "split": split,
                "shape": list(maze.grid.shape),
                "encoding": "uint8; 0=free, 1=occupied",
            }
        )
        for _ in range(sample_count):
            x, y, theta = _sample_pose(maze, rng)
            scan = cast_lidar(maze, Agent(x, y, theta), cfg)
            map_ids[row] = map_id
            scans[row] = scan.ranges
            hit_mask[row] = scan.hit_mask
            poses[row] = (x, y, np.cos(theta), np.sin(theta))
            maze_names[row] = maze.name
            splits[row] = split
            row += 1

    relative_angles = np.linspace(
        -cfg.fov / 2.0, cfg.fov / 2.0, cfg.n_rays, endpoint=False
    )
    metadata = {
        "dataset_version": DATASET_VERSION,
        "schema": {
            "scans": "[sample, beam], float32 range in maze-cell units",
            "hit_mask": "[sample, beam], bool; false means max-range/no hit",
            "maps": "[map, row, col], uint8 occupancy grid; 0=free, 1=occupied",
            "map_ids": "[sample], int32 index into maps for the model input",
            "poses": "[sample, 4], float32 [x_cells, y_cells, cos(theta), sin(theta)]",
            "maze_names": "per-sample map name; provenance metadata",
            "splits": "whole maps held out to prevent layout leakage",
        },
        "lidar": {
            "n_rays": cfg.n_rays,
            "fov_radians": float(cfg.fov),
            "max_range_cells": float(cfg.max_range),
            "ray_step_cells": float(cfg.step),
            "relative_beam_angles_radians": relative_angles.tolist(),
            "beam_order": "increasing angle from -fov/2 to +fov/2, endpoint excluded",
            "heading_convention": "0 is +x; positive turns toward +y (down in map rows)",
            "range_semantics": "wall distance, or max_range when hit_mask is false",
            "noise_model": "none",
        },
        "pose_sampling": "uniform free cell, uniform position within inner 60%, uniform heading",
        "learning_task": (
            "input: maps[map_ids[i]] and scans[i]; target: poses[i] in map coordinates"
        ),
        "samples_per_maze": samples_per_maze,
        "random_rooms_per_split": random_rooms_per_split,
        "samples_per_random_room": samples_per_random_room,
        "seed": seed,
        "split_mazes": {
            "train": list(TRAIN_MAZES),
            "validation": list(VALIDATION_MAZES),
            "test": list(TEST_MAZES),
        },
        "random_room_ids": random_room_ids,
        "maps": map_records,
    }
    return {
        "maps": maps,
        "map_ids": map_ids,
        "scans": scans,
        "hit_mask": hit_mask,
        "poses": poses,
        "maze_names": maze_names,
        "splits": splits,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def save_dataset(dataset: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    return path


def inspect_dataset(
    dataset_path: str | Path,
    plot_path: str | Path | None = None,
    sample_index: int = 0,
) -> None:
    """Print schema/range checks and optionally save a polar scan preview."""
    with np.load(dataset_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        scans = data["scans"]
        hit_mask = data["hit_mask"]
        maps = data["maps"]
        map_ids = data["map_ids"]
        poses = data["poses"]
        maze_names = data["maze_names"]
        splits = data["splits"]

        if scans.ndim != 2 or hit_mask.shape != scans.shape:
            raise ValueError("Expected 2D scans and a same-shaped hit_mask")
        if maps.ndim != 3 or maps.shape[1:] != (41, 41):
            raise ValueError("Expected maps shaped [map, 41, 41]")
        if map_ids.shape != (len(scans),):
            raise ValueError("Expected one map_id per scan")
        if np.any(map_ids < 0) or np.any(map_ids >= len(maps)):
            raise ValueError("Dataset contains an invalid map_id")
        if poses.shape != (len(scans), len(POSE_LABELS)):
            raise ValueError(f"Expected pose labels shaped (N, {len(POSE_LABELS)})")
        if len(maze_names) != len(scans) or len(splits) != len(scans):
            raise ValueError("Each scan must have a maze name and split")
        if not np.isfinite(scans).all() or not np.isfinite(poses).all():
            raise ValueError("Dataset contains non-finite scans or pose labels")
        if not 0 <= sample_index < len(scans):
            raise IndexError(
                f"sample_index must be between 0 and {len(scans) - 1}"
            )

        print(f"Dataset version: {metadata['dataset_version']}")
        print(
            f"Samples: {len(scans)}; unique maps: {len(maps)}; "
            f"scan shape: {scans.shape}; pose shape: {poses.shape}"
        )
        print(
            f"Selected sample map: {maze_names[sample_index]} "
            f"(map_id={int(map_ids[sample_index])}, split={splits[sample_index]})"
        )
        print(f"Range: {scans.min():.2f}–{scans.max():.2f} maze cells")
        print(f"Max range: {metadata['lidar']['max_range_cells']:.2f} cells")
        for split in ("train", "validation", "test"):
            mask = splits == split
            print(
                f"{split}: {int(mask.sum())} samples; "
                f"mazes={', '.join(np.unique(maze_names[mask]).tolist()) or 'none'}"
            )
        print(f"Wall returns: {int(hit_mask.sum())}; max-range returns: {int((~hit_mask).sum())}")

        if plot_path is None:
            return

        import matplotlib.pyplot as plt

        relative_angles = np.asarray(
            metadata["lidar"]["relative_beam_angles_radians"], dtype=np.float64
        )
        pose = poses[sample_index]
        theta = float(np.arctan2(pose[3], pose[2]))
        fig = plt.figure(figsize=(12, 6))
        map_ax = fig.add_subplot(1, 2, 1)
        scan_ax = fig.add_subplot(1, 2, 2, projection="polar")
        map_id = int(map_ids[sample_index])
        occupancy = maps[map_id]
        map_ax.imshow(occupancy, cmap="gray_r", origin="upper", interpolation="nearest")
        map_ax.scatter(pose[0], pose[1], marker="*", s=100, c="#ff3b30", label="true pose")
        map_ax.set_xlim(-0.5, occupancy.shape[1] - 0.5)
        map_ax.set_ylim(occupancy.shape[0] - 0.5, -0.5)
        map_ax.set_aspect("equal")
        map_ax.set_title(f"Map input — {maze_names[sample_index]}")

        beam_angles = relative_angles + theta
        valid_hits = hit_mask[sample_index]
        hit_x = pose[0] + scans[sample_index] * np.cos(beam_angles)
        hit_y = pose[1] + scans[sample_index] * np.sin(beam_angles)
        map_ax.scatter(
            hit_x[valid_hits],
            hit_y[valid_hits],
            s=8,
            c="#23a6d5",
            label="LiDAR returns",
        )
        map_ax.legend(loc="upper right")
        scan_ax.set_theta_zero_location("N")
        scan_ax.set_theta_direction(-1)
        scan_ax.scatter(relative_angles, scans[sample_index], s=12, c="#23a6d5")
        scan_ax.set_title(
            f"LiDAR scan — {maze_names[sample_index]}\n"
            f"sample {sample_index}, pose=({pose[0]:.1f}, {pose[1]:.1f}), "
            f"heading={np.degrees(theta):.1f}°"
        )
        path = Path(plot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved map-and-scan preview: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="data/localization_v3.npz",
        help="compressed dataset output path (default: %(default)s)",
    )
    parser.add_argument(
        "--samples-per-maze", type=int, default=500,
        help="number of random poses to sample in each maze (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random-rooms-per-split",
        type=int,
        default=5,
        help="independent generated rooms for each split (default: %(default)s)",
    )
    parser.add_argument(
        "--samples-per-random-room",
        type=int,
        default=100,
        help="poses sampled in each generated room (default: %(default)s)",
    )
    parser.add_argument(
        "--inspect", metavar="DATASET",
        help="inspect an existing dataset instead of generating one",
    )
    parser.add_argument(
        "--plot", metavar="PNG",
        help="save a polar plot of the selected scan (use with --inspect)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="sample index to inspect or plot (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(args.inspect, args.plot, args.sample)
        return
    if args.plot:
        parser.error("--plot requires --inspect")
    dataset = generate_dataset(
        samples_per_maze=args.samples_per_maze,
        seed=args.seed,
        random_rooms_per_split=args.random_rooms_per_split,
        samples_per_random_room=args.samples_per_random_room,
    )
    path = save_dataset(dataset, args.output)
    print(f"Generated {len(dataset['scans'])} scans: {path}")
    print(
        f"Inspect with: python -m sim.dataset --inspect {path} "
        "--sample 0 --plot data/localization_preview_v3.png"
    )


if __name__ == "__main__":
    main()
