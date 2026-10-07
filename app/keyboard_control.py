"""Local Streamlit component: capture WASD / arrow keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent / "keyboard_component"

_keyboard = components.declare_component(
    "ghost_keyboard",
    path=str(_COMPONENT_DIR),
)


def keyboard_control(enabled: bool = True, key: Optional[str] = None) -> Any:
    """Return ``{"key": "up"|"down"|"left"|"right", "t": ...}`` or None."""
    if not enabled:
        return None
    return _keyboard(default=None, key=key)
