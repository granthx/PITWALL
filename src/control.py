"""Driving-policy helpers shared by the simulator bridge (src.drive) and the
web console (src.webapp) — one source of truth for control math."""

import cv2
import numpy as np

from .config import CFG


def throttle_for(speed: float, steering: float, max_speed: float | None = None) -> float:
    """Slow down proactively in turns, brake if overspeeding into a corner,
    and provide strong torque on straights / uphills."""
    top = max_speed if max_speed is not None else CFG.max_speed
    turn_severity = min(abs(steering) * 3.0, 1.0)
    target = top - (top - CFG.min_speed) * turn_severity
    if speed > target + 1.5 and abs(steering) > 0.12:
        # Active corner-entry braking so high-speed runs don't drift wide
        return float(np.clip((target - speed) * 0.25, -0.8, 0.0))
    return float(np.clip(1.0 - speed / max(target, 1.0), -0.5, 1.0))


def smooth_steering(new: float, prev: float, weight: float) -> float:
    """Low-pass the steering: s = weight*s_prev + (1-weight)*s_new.

    weight=0 -> passthrough; 0.2-0.5 tames zig-zag on the track.
    """
    s = weight * prev + (1.0 - weight) * new
    return float(np.clip(s, -1.0, 1.0))


def lane_centering_assist(rgb: np.ndarray) -> float:
    """Vision-based boundary repulsion & road-symmetry guard."""
    roi = rgb[95:135, :]
    r = roi[:, :, 0].astype(np.float32)
    g = roi[:, :, 1].astype(np.float32)
    b = roi[:, :, 2].astype(np.float32)

    # 1. Red curb repulsion (Track 1 red/white boundary stripes)
    red_curb = ((r > 150) & ((r - g) > 40) & ((r - b) > 40)).astype(np.float32)
    left_curb = float(red_curb[:, :115].mean())
    right_curb = float(red_curb[:, 205:].mean())
    curb_nudge = (left_curb - right_curb) * 1.4

    # 2. Asphalt road-surface symmetry
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    asphalt = ((sat < 55) & (val > 30) & (val < 215) & (red_curb < 0.5)).astype(np.float32)
    left_road = float(asphalt[:, 15:120].mean())
    right_road = float(asphalt[:, 200:305].mean())
    road_nudge = (right_road - left_road) * 0.30

    return float(np.clip(curb_nudge + road_nudge, -0.35, 0.35))
