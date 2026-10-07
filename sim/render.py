"""Matplotlib figures: god-mode full map vs ghost-mode LiDAR-only map."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Patch

from .lidar import LidarConfig, LidarScan, cast_lidar
from .maze import Agent, Maze

ZONE_COLORS = {
    "symmetric": "#6aa6ff",
    "sparse": "#f0c14a",
    "rich": "#7dcea0",
}

# explored: -1 unknown, 0 free, 1 occupied
UNKNOWN, FREE, OCCUPIED = -1, 0, 1


def blank_explored(maze: Maze) -> np.ndarray:
    return np.full((maze.height, maze.width), UNKNOWN, dtype=np.int8)


def integrate_scan(
    explored: np.ndarray,
    maze: Maze,
    agent: Agent,
    scan: LidarScan,
    step: float = 0.05,
) -> np.ndarray:
    """Carve free space along rays and mark hit cells occupied (ghost map memory)."""
    out = explored.copy()
    for ang, rng in zip(scan.angles, scan.ranges):
        c, s = float(np.cos(ang)), float(np.sin(ang))
        dist = 0.0
        while dist < rng - step:
            dist += step
            x = agent.x + c * dist
            y = agent.y + s * dist
            if not maze.in_bounds(x, y):
                break
            col, row = maze.cell(x, y)
            if out[row, col] != OCCUPIED:
                out[row, col] = FREE
        hx = agent.x + c * rng
        hy = agent.y + s * rng
        if maze.in_bounds(hx, hy):
            col, row = maze.cell(hx, hy)
            out[row, col] = OCCUPIED
    # mark agent cell free
    if maze.in_bounds(agent.x, agent.y):
        col, row = maze.cell(agent.x, agent.y)
        out[row, col] = FREE
    return out


def _zone_overlay(maze: Maze) -> np.ndarray:
    overlay = np.zeros_like(maze.grid, dtype=np.int8)
    order = {"symmetric": 1, "sparse": 2, "rich": 3}
    for name, (c0, r0, c1, r1) in maze.zones.items():
        overlay[r0 : r1 + 1, c0 : c1 + 1] = order[name]
    return overlay


def _draw_agent(ax, agent: Agent, color: str, glow: bool = False) -> None:
    if glow:
        ax.plot(agent.x, agent.y, "o", color=color, markersize=18, alpha=0.25, zorder=3)
        ax.plot(agent.x, agent.y, "o", color=color, markersize=12, alpha=0.45, zorder=4)
    ax.add_patch(
        FancyArrow(
            agent.x,
            agent.y,
            0.7 * np.cos(agent.theta),
            0.7 * np.sin(agent.theta),
            width=0.12,
            head_width=0.45,
            head_length=0.35,
            length_includes_head=True,
            color=color,
            zorder=5,
        )
    )
    ax.plot(agent.x, agent.y, "o", color=color, markersize=6, zorder=6)


def _style_map_ax(ax, maze: Maze, title: str) -> None:
    ax.set_title(title)
    ax.set_xlim(-0.5, maze.width - 0.5)
    ax.set_ylim(maze.height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def figure_god_and_ghost(
    maze: Maze,
    agent: Agent,
    scan: LidarScan,
    explored: Optional[np.ndarray] = None,
    ghost_pose: Optional[Agent] = None,
    show_rays_on_god: bool = True,
    figsize: Tuple[float, float] = (16.0, 8.0),
) -> plt.Figure:
    """
    Side-by-side:
      God mode  — full known maze (whole area).
      Ghost mode — darkness; only LiDAR-explored free/occupied cells + glowing figure.
    Until a CNN exists, ghost_pose defaults to the true agent (layout preview).
    """
    if explored is None:
        explored = integrate_scan(blank_explored(maze), maze, agent, scan)
    if ghost_pose is None:
        ghost_pose = agent

    fig, (ax_god, ax_ghost) = plt.subplots(1, 2, figsize=figsize)

    # ----- God mode: whole area -----
    overlay = _zone_overlay(maze)
    zone_cmap = ListedColormap(["#ffffff00", "#6aa6ff55", "#f0c14a55", "#7dcea055"])
    ax_god.set_facecolor("#f4f4f4")
    ax_god.imshow(overlay, cmap=zone_cmap, origin="upper", vmin=0, vmax=3, interpolation="nearest")
    wall_cmap = ListedColormap(["#00000000", "#1b1b1b"])
    ax_god.imshow(maze.grid, cmap=wall_cmap, origin="upper", vmin=0, vmax=1, interpolation="nearest")

    if show_rays_on_god:
        for hx, hy in zip(scan.hit_x, scan.hit_y):
            ax_god.plot([agent.x, hx], [agent.y, hy], color="#e74c3c55", linewidth=0.5, zorder=2)
        ax_god.scatter(scan.hit_x, scan.hit_y, s=5, c="#c0392b", zorder=3)

    _draw_agent(ax_god, agent, color="#1a73e8", glow=False)

    zone = maze.zone_at(agent.x, agent.y)
    god_title = "God mode — full map"
    if zone:
        god_title += f"  ·  {zone}"
    _style_map_ax(ax_god, maze, god_title)
    ax_god.legend(
        handles=[
            Patch(facecolor=c, alpha=0.55, label=n) for n, c in ZONE_COLORS.items()
        ]
        + [Line2D([0], [0], marker="o", color="#1a73e8", linestyle="None", label="true pose")],
        loc="upper right",
        fontsize=7,
        framealpha=0.9,
    )

    # ----- Ghost mode: LiDAR-built map only -----
    ax_ghost.set_facecolor("#050508")
    # paint explored: unknown stays black, free dim, occupied bright
    ghost_rgb = np.zeros((maze.height, maze.width, 3), dtype=np.float64)
    free_mask = explored == FREE
    occ_mask = explored == OCCUPIED
    ghost_rgb[free_mask] = (0.12, 0.16, 0.28)
    ghost_rgb[occ_mask] = (0.85, 0.9, 1.0)
    ax_ghost.imshow(ghost_rgb, origin="upper", interpolation="nearest")

    # current scan hits glow
    ax_ghost.scatter(scan.hit_x, scan.hit_y, s=10, c="#7fdbff", alpha=0.9, zorder=3)

    _draw_agent(ax_ghost, ghost_pose, color="#b388ff", glow=True)

    _style_map_ax(ax_ghost, maze, "Ghost mode — LiDAR map only")
    ax_ghost.legend(
        handles=[
            Patch(facecolor=(0.12, 0.16, 0.28), label="seen free"),
            Patch(facecolor=(0.85, 0.9, 1.0), label="hit / wall"),
            Line2D([0], [0], marker="o", color="#b388ff", linestyle="None", label="ghost pose"),
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
        labelcolor="white",
        facecolor="#1a1a22",
        edgecolor="#333",
    )
    ax_ghost.tick_params(colors="#aaa")
    ax_ghost.xaxis.label.set_color("#aaa")
    ax_ghost.yaxis.label.set_color("#aaa")
    ax_ghost.title.set_color("#ddd")
    for spine in ax_ghost.spines.values():
        spine.set_color("#333")

    fig.tight_layout()
    return fig


# Keep thin wrappers for any older callers / debugging
def figure_truth(maze: Maze, agent: Agent, scan: Optional[LidarScan] = None, **kwargs):
    if scan is None:
        scan = cast_lidar(maze, agent, LidarConfig())
    explored = integrate_scan(blank_explored(maze), maze, agent, scan)
    return figure_god_and_ghost(maze, agent, scan, explored=explored, **kwargs)
