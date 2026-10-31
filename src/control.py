"""Driving-policy helpers shared by the simulator bridge (src.drive) and the
web console (src.webapp) — one source of truth for control math."""

import numpy as np

from .config import CFG


def throttle_for(speed: float, steering: float) -> float:
    """Slow down in sharp turns, speed up on straights (simple P-style rule)."""
    target = CFG.max_speed - (CFG.max_speed - CFG.min_speed) * min(abs(steering) * 2.0, 1.0)
    return float(np.clip(1.0 - speed / target, -0.5, 1.0))


def smooth_steering(new: float, prev: float, weight: float) -> float:
    """Low-pass the steering: s = weight*s_prev + (1-weight)*s_new.

    weight=0 -> passthrough; 0.5-0.8 tames zig-zag on the track.
    """
    s = weight * prev + (1.0 - weight) * new
    return float(np.clip(s, -1.0, 1.0))
