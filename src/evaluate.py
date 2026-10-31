"""Evaluate a trained steering model on held-out driving data.

    python -m src.evaluate --model models/pilotnet.pt --data data/recording

Reports MAE / RMSE in normalised units AND degrees (x sim_max_angle_deg), plus
a predicted-vs-human scatter plot. By default outputs are written next to the
model checkpoint (eval.json / eval_scatter.png) so they travel with the run.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CFG
from .data import DrivingDataset, expand_cameras, load_log
from .model import build_model


def evaluate(model, samples, device="cpu", batch_size=128, workers=0):
    """Run inference over (path, angle) samples. Returns (ys, ps) numpy arrays."""
    dl = DataLoader(DrivingDataset(samples, training=False), batch_size=batch_size,
                    shuffle=False, num_workers=workers)
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            ps.append(model(x.to(device)).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys).ravel(), np.concatenate(ps).ravel()


def metrics(ys: np.ndarray, ps: np.ndarray) -> dict:
    err = ps - ys
    mae = float(np.abs(err).mean())
    return {
        "n": int(len(ys)),
        "mae": mae,
        "mae_deg": mae * CFG.sim_max_angle_deg,
        "rmse": float(np.sqrt((err ** 2).mean())),
        "rmse_deg": float(np.sqrt((err ** 2).mean()) * CFG.sim_max_angle_deg),
        "max_abs_err": float(np.abs(err).max()),
        "max_abs_err_deg": float(np.abs(err).max()) * CFG.sim_max_angle_deg,
        "bias": float(err.mean()),
        "bias_deg": float(err.mean()) * CFG.sim_max_angle_deg,
    }


def plot_scatter(ys, ps, path, title="predicted vs human steering"):
    deg = CFG.sim_max_angle_deg
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(ys * deg, ps * deg, s=6, alpha=0.45, color="#1f77b4", edgecolors="none")
    lim = deg * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="perfect")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("human steering (deg)")
    ax.set_ylabel("model guess (deg)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="folder with driving_log.csv and IMG/")
    ap.add_argument("--n", type=int, default=5000, help="max centre-camera rows to evaluate")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=None, help="base path for eval.json/eval_scatter.png "
                                                "(default: next to the model checkpoint)")
    a = ap.parse_args()

    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    ckpt = torch.load(a.model, map_location=device)
    model = build_model(ckpt.get("arch", "pilotnet")).to(device)
    model.load_state_dict(ckpt["state_dict"])

    df = load_log(a.data).head(a.n)
    samples = expand_cameras(df, a.data)[::3]  # centre camera only, no offsets
    ys, ps = evaluate(model, samples, device=device, batch_size=a.batch_size, workers=a.workers)
    m = metrics(ys, ps)
    m.update({"model": a.model, "data": a.data,
              "val_mse_ckpt": float(ckpt.get("val_mse", float("nan")))})

    base = a.out or os.path.splitext(a.model)[0]
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    with open(base + "_eval.json", "w") as f:
        json.dump(m, f, indent=2)
    plot_scatter(ys, ps, base + "_eval.png",
                 title=f"MAE {m['mae_deg']:.2f} deg (n={m['n']})")
    print(json.dumps(m, indent=2))
    print(f"saved -> {base}_eval.json, {base}_eval.png")


if __name__ == "__main__":
    main()
