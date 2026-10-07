"""Streamlit UI: god-mode full map + ghost-mode LiDAR map, side by side."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.keyboard_control import keyboard_control
from sim.lidar import LidarConfig, cast_lidar
from sim.maze import MAZE_SIZE, Agent, Maze
from sim.render import blank_explored, figure_god_and_ghost, integrate_scan

STEP = 0.5
TURN = 0.25  # radians

KEY_ACTIONS = {
    "up": (STEP, 0.0),
    "down": (-STEP, 0.0),
    "left": (0.0, -TURN),
    "right": (0.0, TURN),
}


def _init_state() -> None:
    need_maze = (
        "maze" not in st.session_state
        or st.session_state.maze.grid.shape != (MAZE_SIZE, MAZE_SIZE)
    )
    if need_maze:
        st.session_state.maze = Maze.teaching_maze()
        start = st.session_state.maze.start
        st.session_state.agent = Agent(start.x, start.y, start.theta)
        st.session_state.explored = blank_explored(st.session_state.maze)
    if "agent" not in st.session_state:
        start = st.session_state.maze.start
        st.session_state.agent = Agent(start.x, start.y, start.theta)
    if "explored" not in st.session_state:
        st.session_state.explored = blank_explored(st.session_state.maze)
    if "keyboard_enabled" not in st.session_state:
        st.session_state.keyboard_enabled = False
    if "last_key_t" not in st.session_state:
        st.session_state.last_key_t = 0


def _move(forward: float, dtheta: float) -> None:
    maze: Maze = st.session_state.maze
    st.session_state.agent = maze.try_move(st.session_state.agent, forward, dtheta)


def _apply_keyboard_event(event) -> None:
    if not event or not isinstance(event, dict):
        return
    name = event.get("key")
    t = event.get("t", 0)
    if name not in KEY_ACTIONS:
        return
    if t and t == st.session_state.last_key_t:
        return
    st.session_state.last_key_t = t
    forward, dtheta = KEY_ACTIONS[name]
    _move(forward, dtheta)


st.set_page_config(page_title="Ghost Localize — God vs Ghost", layout="wide")
st.title("Ghost Localize")
st.caption(
    "God mode shows the whole maze. Ghost mode shows only what LiDAR has revealed "
    "(CNN pose estimate comes later — ghost marker tracks truth for now)."
)

_init_state()
maze: Maze = st.session_state.maze

with st.sidebar:
    st.header("Controls")

    if st.button(
        "Disable keyboard control"
        if st.session_state.keyboard_enabled
        else "Enable keyboard control",
        use_container_width=True,
        type="primary" if not st.session_state.keyboard_enabled else "secondary",
    ):
        st.session_state.keyboard_enabled = not st.session_state.keyboard_enabled
        st.rerun()

    if st.session_state.keyboard_enabled:
        st.caption("WASD or arrows · click the green box if keys are ignored")
        event = keyboard_control(enabled=True, key="ghost_kb")
        _apply_keyboard_event(event)
    else:
        st.caption("Click **Enable keyboard control** to drive with the keyboard.")

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⬅", use_container_width=True):
            _move(0.0, -TURN)
    with c2:
        if st.button("⬆", use_container_width=True):
            _move(STEP, 0.0)
    with c3:
        if st.button("➡", use_container_width=True):
            _move(0.0, TURN)
    if st.button("⬇ back", use_container_width=True):
        _move(-STEP, 0.0)
    if st.button("Reset pose", use_container_width=True):
        s = maze.start
        st.session_state.agent = Agent(s.x, s.y, s.theta)
    if st.button("Clear ghost map", use_container_width=True):
        st.session_state.explored = blank_explored(maze)

    st.divider()
    st.header("LiDAR")
    n_rays = st.slider("Rays", 36, 360, 180, 6)
    max_range = st.slider("Max range", 6.0, 40.0, 20.0, 0.5)
    show_rays = st.checkbox("Draw rays on god map", value=True)

    agent = st.session_state.agent
    st.divider()
    zone = maze.zone_at(agent.x, agent.y) or "—"
    st.metric("Zone", zone)
    st.write(f"pose: x={agent.x:.2f}, y={agent.y:.2f}, θ={agent.theta:.2f} rad")

agent = st.session_state.agent
cfg = LidarConfig(n_rays=n_rays, max_range=max_range)
scan = cast_lidar(maze, agent, cfg)
st.session_state.explored = integrate_scan(st.session_state.explored, maze, agent, scan)

fig = figure_god_and_ghost(
    maze,
    agent,
    scan,
    explored=st.session_state.explored,
    ghost_pose=agent,
    show_rays_on_god=show_rays,
)
st.pyplot(fig, clear_figure=True)
plt.close("all")

st.info(
    "**Left (god):** full area, walls, teaching zones, true pose. "
    "**Right (ghost):** same canvas in the dark — only free/hit cells from LiDAR accumulate as you move."
)
