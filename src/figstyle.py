"""Shared matplotlib style: the dark-navy console theme used by every figure
this project writes (loss curves, histograms, maps, saliency, story, eval)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANEL = "#0A1424"  # figure background
INK = "#D7E4F2"  # primary text
DIM = "#7E93AC"  # secondary text / labels
GRID = "#22334C"  # grid lines / spines
TEAL = "#3EE6C1"  # model / accent
AMBER = "#F5A94B"  # human / warning accent
CORAL = "#FF6E5E"  # error
BLUE = "#6EA8FF"  # info


def navy_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PANEL,
            "axes.facecolor": PANEL,
            "savefig.facecolor": PANEL,
            "text.color": INK,
            "axes.edgecolor": GRID,
            "axes.labelcolor": DIM,
            "xtick.color": DIM,
            "ytick.color": DIM,
            "axes.titlecolor": INK,
            "font.family": ["DejaVu Sans Mono"],
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=[TEAL, AMBER, BLUE, CORAL]),
        }
    )


def saliency_lut() -> np.ndarray:
    """256x3 uint8 LUT: dark navy -> blue -> teal -> amber -> warm white."""
    stops = [
        (0.00, (8, 18, 38)),
        (0.35, (18, 80, 120)),
        (0.60, (62, 230, 193)),
        (0.82, (245, 169, 75)),
        (1.00, (255, 241, 214)),
    ]
    xs = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 256)
    lut = np.stack([np.interp(t, xs, cols[:, k]) for k in range(3)], axis=1)
    return np.clip(lut, 0, 255).astype(np.uint8)
