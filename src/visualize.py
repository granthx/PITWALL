"""Visual tools matching the video: steering histogram, feature maps
(lines -> shapes), and predicted-vs-actual angle plots.

    python -m src.visualize hist  --data data/recording
    python -m src.visualize maps  --model models/pilotnet.pt --image data/recording/IMG/<file>.jpg
    python -m src.visualize eval  --model models/pilotnet.pt --data data/recording
"""
import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import CFG
from .data import balance, expand_cameras, load_log
from .model import PilotNet
from .preprocess import preprocess, to_tensor_array


def cmd_hist(a):
    df = load_log(a.data)
    bal = balance(df)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5), sharey=False)
    ax[0].hist(df["steering"], bins=CFG.n_hist_bins); ax[0].set_title("raw steering")
    ax[1].hist(bal["steering"], bins=CFG.n_hist_bins); ax[1].set_title("balanced steering")
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("saved", a.out)


def cmd_maps(a):
    """Show what each conv layer 'sees': early = edges/lines, later = shapes."""
    model = PilotNet(); model.load_state_dict(torch.load(a.model, map_location="cpu")["state_dict"]); model.eval()
    rgb = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(to_tensor_array(preprocess(rgb))).unsqueeze(0)
    convs = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]
    acts, h, i = [], x / 127.5 - 1.0, 0
    for layer in model.features:
        h = layer(h)
        if isinstance(layer, torch.nn.ELU):
            acts.append(h.detach()[0].mean(0).numpy()); i += 1
    fig, ax = plt.subplots(1, len(acts) + 1, figsize=(3 * (len(acts) + 1), 2.6))
    ax[0].imshow(rgb); ax[0].set_title("input")
    for k, m in enumerate(acts, 1):
        ax[k].imshow(m, cmap="viridis"); ax[k].set_title(f"conv{k} mean map")
    for x_ in ax: x_.axis("off")
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("saved", a.out)


def cmd_eval(a):
    from .data import DrivingDataset
    model = PilotNet(); model.load_state_dict(torch.load(a.model, map_location="cpu")["state_dict"]); model.eval()
    samples = expand_cameras(load_log(a.data).head(a.n), a.data)[::3]  # centre cams only
    ds = DrivingDataset(samples, training=False)
    ys, ps = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]; ps.append(model(x[None]).item()); ys.append(y.item())
    ys, ps = np.array(ys), np.array(ps)
    print(f"MAE: {np.abs(ys - ps).mean():.4f}  ({np.abs(ys - ps).mean() * CFG.sim_max_angle_deg:.2f} deg)")
    plt.figure(figsize=(5, 5)); plt.scatter(ys, ps, s=6, alpha=.5)
    plt.plot([-1, 1], [-1, 1], "r--"); plt.xlabel("human (yours)"); plt.ylabel("model guess")
    plt.savefig(a.out, dpi=130); print("saved", a.out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    h = sp.add_parser("hist"); h.add_argument("--data", required=True); h.add_argument("--out", default="runs/hist.png")
    m = sp.add_parser("maps"); m.add_argument("--model", required=True); m.add_argument("--image", required=True); m.add_argument("--out", default="runs/maps.png")
    e = sp.add_parser("eval"); e.add_argument("--model", required=True); e.add_argument("--data", required=True); e.add_argument("--n", type=int, default=2000); e.add_argument("--out", default="runs/eval.png")
    a = ap.parse_args()
    import os; os.makedirs("runs", exist_ok=True)
    {"hist": cmd_hist, "maps": cmd_maps, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
