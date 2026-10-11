"""Local Streamlit component: capture WASD / arrow keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import streamlit.components.v1 as components

from sim.lidar import LidarConfig
from sim.maze import Agent, Maze

_COMPONENT_DIR = Path(__file__).parent / "keyboard_component"

_keyboard = components.declare_component(
    "ghost_keyboard",
    path=str(_COMPONENT_DIR),
)


def keyboard_control(
    maze: Maze,
    agent: Agent,
    explored: np.ndarray,
    config: LidarConfig,
    enabled: bool,
    show_rays: bool,
    pose_revision: int,
    map_revision: int,
    key: Optional[str] = None,
) -> Any:
    """Render the live local map and return periodic browser pose/map snapshots."""
    return _keyboard(
        default=None,
        key=key,
        grid=maze.grid.tolist(),
        zones=maze.zones,
        agent={"x": agent.x, "y": agent.y, "theta": agent.theta},
        explored=explored.tolist(),
        n_rays=config.n_rays,
        max_range=config.max_range,
        enabled=enabled,
        show_rays=show_rays,
        pose_revision=pose_revision,
        map_revision=map_revision,
    )
