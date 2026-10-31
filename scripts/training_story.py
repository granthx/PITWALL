"""Reproduce the video's "guess -> error -> adjust" story for one fixed image.

For a single frame, show the model's steering prediction with a RANDOM init
(before any training) and after every epoch, next to the human steering angle
and the error. Requires a run directory with per-epoch checkpoints (written by
default by `src.train`).

    python scripts/training_story.py --run runs/<timestamp> --data data/synthetic
    python scripts/training_story.py --run runs/<timestamp> \
        --image data/synthetic/IMG/center_00042.jpg

Outputs: a console table and runs/training_story.png (or --out).
"""

import argparse
import glob
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.data import COLUMNS, load_log
from src.figstyle import AMBER, DIM, INK, TEAL, navy_style
from src.model import build_model
from src.preprocess import preprocess, to_tensor_array


def find_run_dir(run):
    """Explicit --run, else the newest run dir that has per-epoch checkpoints."""
    if run:
        return run
    cands = sorted(glob.glob("runs/*/epoch_*.pt"))
    if not cands:
        sys.exit(
            "no per-epoch checkpoints found; train first:\n"
            "  python -m src.train --data <data-dir> --epochs 15"
        )
    return os.path.dirname(cands[-1])


def human_angle(data_dir: str, image: str) -> float:
    """Look up the steering label for an image path (any camera) in the CSV."""
    df = load_log(data_dir)
    base = os.path.basename(image)
    for _, r in df.iterrows():
        for col, off in zip(COLUMNS[:3], (0.0, CFG.steering_correction, -CFG.steering_correction)):
            if os.path.basename(str(r[col])) == base:
                return float(np.clip(r["steering"] + off, -1, 1))
    sys.exit(f"'{base}' not found in {os.path.join(data_dir, 'driving_log.csv')}")


def predict(model, rgb: np.ndarray) -> float:
    x = torch.from_numpy(to_tensor_array(preprocess(rgb))).unsqueeze(0)
    with torch.no_grad():
        return float(model(x).item())


def load_epochs(run_dir: str):
    eps = sorted(glob.glob(os.path.join(run_dir, "epoch_*.pt")))
    if not eps:
        sys.exit(f"no epoch_XX.pt files in {run_dir} (train without --no-epoch-ckpts)")
    arch = torch.load(eps[0], map_location="cpu").get("arch", "pilotnet")
    return [(int(os.path.basename(p).split("_")[1].split(".")[0]), p, arch) for p in eps]


def draw(rgb, rows, out, deg=CFG.sim_max_angle_deg):
    """rows: list of (label, guess_norm, human_norm)."""
    navy_style()
    n = len(rows)
    fig = plt.figure(figsize=(9.5, 0.62 * n + 3.2))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.35, 2.0], wspace=0.18, left=0.045, right=0.97, top=0.86, bottom=0.10
    )

    aximg = fig.add_subplot(gs[0, 0])
    aximg.imshow(rgb)
    aximg.set_title("the one image it sees", color=INK, fontsize=10)
    aximg.set_xticks([]), aximg.set_yticks([])
    for s in aximg.spines.values():
        s.set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    ys = np.arange(n)[::-1]
    for (_label, guess, human), y in zip(rows, ys):
        ax.plot([guess * deg], [y], marker="o", ms=8, color=TEAL, zorder=3)
        ax.plot([human * deg], [y], marker="|", ms=22, mew=2.6, color=AMBER, zorder=3)
        ax.plot([-deg, deg], [y, y], color="#22334C", lw=1, zorder=1)
        err = abs(guess - human) * deg
        ax.annotate(
            f"err {err:5.1f}°",
            (deg, y),
            xytext=(6, -4),
            textcoords="offset points",
            color=DIM,
            fontsize=8,
        )
    ax.axvline(0, color=DIM, lw=0.8, ls="--", alpha=0.6)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(-deg, deg * 1.45)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("steering (deg)   ·   ○ model guess   | human", color=DIM, fontsize=9)
    ax.set_title("guess → error → adjust", color=INK, fontsize=11)

    fig.suptitle(
        "TRAINING STORY — one image, epoch after epoch", color=INK, fontsize=12, x=0.045, ha="left"
    )
    fig.text(
        0.045,
        0.935,
        f"amber | = human · teal ○ = model · error in degrees · {out}",
        color=DIM,
        fontsize=8,
    )
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir with epoch_XX.pt (default: latest)")
    ap.add_argument("--data", default=None, help="data dir with driving_log.csv (for the label)")
    ap.add_argument("--image", default=None, help="frame to follow (default: a turning frame)")
    ap.add_argument("--out", default="runs/training_story.png")
    a = ap.parse_args()

    run_dir = find_run_dir(a.run)
    data_dir = a.data
    if data_dir is None:
        # fall back to the folder recorded in the run's config snapshot
        cfg_path = os.path.join(run_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                data_dir = json.load(f)["args"]["data"]
    if not data_dir:
        sys.exit("--data is required when the run dir has no config.json")

    # default image: a frame with a real turn in it (more interesting than straight)
    if a.image:
        image = a.image
    else:
        df = load_log(data_dir)
        cand = df.loc[(df["steering"].abs() > 0.3)]
        row = (cand if len(cand) else df).iloc[len(cand) // 2 if len(cand) else 0]
        image = os.path.join(data_dir, "IMG", os.path.basename(str(row["center"])))
    rgb = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    human = human_angle(data_dir, image)

    model = build_model("pilotnet")  # rebuilt per-epoch if the run used another arch
    torch.manual_seed(0)
    rows = [("random init", predict(model, rgb), human)]
    for ep, path, arch in load_epochs(run_dir):
        if arch != "pilotnet":
            model = build_model(arch)
        model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        model.eval()
        rows.append((f"epoch {ep:02d}", predict(model, rgb), human))

    deg = CFG.sim_max_angle_deg
    print(f"\nimage: {image}\nhuman: {human * deg:+.1f} deg  ({human:+.3f} normalised)\n")
    print(f"{'epoch':<12}{'guess (deg)':>12}{'guess (norm)':>14}{'error (deg)':>13}")
    for label, guess, hum in rows:
        print(f"{label:<12}{guess * deg:>+12.1f}{guess:>+14.3f}{abs(guess - hum) * deg:>13.1f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    draw(rgb, rows, a.out)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
