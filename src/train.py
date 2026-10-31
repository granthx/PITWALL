"""Train a steering model on recorded driving data.

    python -m src.train --data data/recording --epochs 15
    python -m src.train --data data/synthetic --epochs 3 --workers 0   # smoke

Artifacts land in ``runs/<timestamp>/`` (config snapshot, history JSON,
loss-curve PNG, best/last + per-epoch checkpoints) and the best checkpoint is
also copied to ``--out`` (default models/pilotnet.pt) for ``src.drive``.
"""

import argparse
import dataclasses
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CFG
from .data import DrivingDataset, balance, expand_cameras, load_log
from .model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, loss_fn, device, opt=None):
    """One pass. Returns (mean loss, mean |error| in normalised units)."""
    train = opt is not None
    model.train(train)
    total, abs_err, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(x)
            abs_err += (pred - y).abs().sum().item()
            n += len(x)
    n = max(n, 1)
    return total / n, abs_err / n


def plot_loss_curve(history, path):
    """Simple train/val MSE curve, written into the run directory."""
    eps = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eps, [h["train_mse"] for h in history], marker="o", ms=3, label="train MSE")
    ax.plot(eps, [h["val_mse"] for h in history], marker="o", ms=3, label="val MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalised steering)")
    ax.set_title("training curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with driving_log.csv and IMG/")
    ap.add_argument("--out", default="models/pilotnet.pt", help="best-checkpoint destination")
    ap.add_argument("--arch", default="pilotnet", choices=["pilotnet", "resnet18"])
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--batch-size", type=int, default=CFG.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--workers", type=int, default=CFG.num_workers)
    ap.add_argument("--seed", type=int, default=CFG.seed, help="random seed (default from config)")
    ap.add_argument("--resume", default=None, help="checkpoint (.pt) to resume from")
    ap.add_argument(
        "--no-epoch-ckpts",
        action="store_true",
        help="skip per-epoch snapshots (needed by scripts/training_story.py)",
    )
    ap.add_argument("--run-dir", default=None, help="override the runs/<timestamp> directory")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device} | arch: {args.arch}")

    run_dir = args.run_dir or time.strftime("runs/%Y%m%d-%H%M%S")
    os.makedirs(run_dir, exist_ok=True)

    df = balance(load_log(args.data))
    samples = expand_cameras(df, args.data)
    tr, va = train_test_split(samples, test_size=CFG.val_split, random_state=args.seed)
    print(f"rows after balancing: {len(df)} | samples: train={len(tr)} val={len(va)}")

    tl = DataLoader(
        DrivingDataset(tr, True, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    vl = DataLoader(
        DrivingDataset(va, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = build_model(args.arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=CFG.weight_decay)
    loss_fn = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    start_epoch, best = 1, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best = ckpt.get("best", float("inf"))
        print(f"resumed from {args.resume} at epoch {start_epoch} (best val MSE so far {best:.5f})")

    # Snapshot of everything that defines this run.
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(
            {
                "cfg": dataclasses.asdict(CFG),
                "args": vars(args),
                "n_train": len(tr),
                "n_val": len(va),
                "torch": torch.__version__,
                "device": device,
            },
            f,
            indent=2,
        )

    history, bad = [], 0
    last_path = os.path.join(run_dir, "last.pt")
    for ep in range(start_epoch, args.epochs + 1):
        t, _ = run_epoch(model, tl, loss_fn, device, opt)
        v, v_mae = run_epoch(model, vl, loss_fn, device)
        sched.step(v)
        history.append(
            {
                "epoch": ep,
                "train_mse": t,
                "val_mse": v,
                "val_mae_deg": v_mae * CFG.sim_max_angle_deg,
                "lr": opt.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch {ep:02d}  train_mse={t:.5f}  val_mse={v:.5f}  "
            f"val_MAE≈{v_mae * CFG.sim_max_angle_deg:.2f} deg"
        )

        state = {"state_dict": model.state_dict(), "arch": args.arch, "epoch": ep}
        if not args.no_epoch_ckpts:
            torch.save(state, os.path.join(run_dir, f"epoch_{ep:02d}.pt"))
        if v < best:
            best = v
            torch.save(
                {**state, "val_mse": v, "best": best, "optimizer": opt.state_dict()}, last_path
            )
            torch.save({**state, "val_mse": v}, args.out)
            print(f"  saved best -> {args.out}")
            bad = 0
        else:
            if os.path.exists(last_path):  # keep optimizer state for --resume
                prev = torch.load(last_path, map_location="cpu")
                torch.save(
                    {
                        **state,
                        "val_mse": v,
                        "best": best,
                        "optimizer": prev.get("optimizer", opt.state_dict()),
                    },
                    last_path,
                )
            else:
                torch.save(
                    {**state, "val_mse": v, "best": best, "optimizer": opt.state_dict()}, last_path
                )
            bad += 1
            if bad >= CFG.patience:
                print("early stopping")
                break

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_loss_curve(history, os.path.join(run_dir, "loss_curve.png"))
    print(f"run artifacts -> {run_dir}")


if __name__ == "__main__":
    main()
