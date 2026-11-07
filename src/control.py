"""Driving-policy helpers shared by the simulator bridge (src.drive) and the
web console (src.webapp) — one source of truth for control math."""

import numpy as np

from .config import CFG


def throttle_for(speed: float, steering: float, max_speed: float | None = None) -> float:
    """Slow down proactively in turns, brake if overspeeding into a corner,
    and provide strong torque on straights / uphills."""
    top = max_speed if max_speed is not None else CFG.max_speed
    turn_severity = min(abs(steering) * 3.0, 1.0)
    target = top - (top - CFG.min_speed) * turn_severity
    return float(np.clip(1.0 - speed / max(target, 1.0), -0.5, 1.0))


def smooth_steering(new: float, prev: float, weight: float) -> float:
    s = weight * prev + (1.0 - weight) * new
    return float(np.clip(s, -1.0, 1.0))
