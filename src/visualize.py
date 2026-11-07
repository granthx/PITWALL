"""Visual tools matching the video's story: "lines -> shapes -> one number".

    python -m src.visualize hist     --data data/recording
    python -m src.visualize maps     --model models/pilotnet.pt --image data/recording/IMG/<f>.jpg
    python -m src.visualize saliency --model models/pilotnet.pt --image data/recording/IMG/<f>.jpg
    python -m src.visualize eval     --model models/pilotnet.pt --data data/recording

`maps` renders per-layer feature-map grids (early layers = edges/lines, later
= shapes). `saliency` overlays where the output gradient is largest — what
the steering angle is most sensitive to. The reusable functions
(`activations`, `gradient_saliency`, `overlay_saliency`) are what the web
console calls per request.
"""

import argparse
import os

import cv2
import numpy as np
import torch

from .config import CFG
from .data import balance, expand_cameras, load_log
from .figstyle import BLUE, DIM, INK, TEAL, navy_style, saliency_lut
from .model import build_model
from .preprocess import preprocess, to_tensor_array

CONV_ROLES = ["LINES", "LINES", "SHAPES", "SHAPES", "SHAPE→ANGLE"]


# ---------------------------------------------------------------- reusable --


def load_model_for(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "pilotnet")
    in_channels = ckpt.get("frame_stack", 1) * 3
    if "stem.0.weight" in ckpt.get("state_dict", {}):
        in_channels = ckpt["state_dict"]["stem.0.weight"].shape[1]
    elif "features.0.weight" in ckpt.get("state_dict", {}):
        in_channels = ckpt["state_dict"]["features.0.weight"].shape[1]
    model = build_model(arch, in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _hook_convs(model: torch.nn.Module):
    """Collect activations after each conv block (post-activation)."""
    acts = []

    def hook(_m, _i, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            acts.append(out.detach()[0].cpu())

    handles = []
    if hasattr(model, "layers") and hasattr(model, "stem"):
        # ResNet/SpatialResNet: hook stem and 4 representative layer blocks
        handles.append(model.stem.register_forward_hook(hook))
        n_blocks = len(model.layers)
        step = max(1, n_blocks // 4)
        for i in range(step - 1, n_blocks, step):
            handles.append(model.layers[i].register_forward_hook(hook))
    else:
        for mod in model.modules():
            if isinstance(mod, (torch.nn.ELU, torch.nn.ReLU)) and not isinstance(mod, torch.nn.ReLU6):
                handles.append(mod.register_forward_hook(hook))
    return acts, handles


def activations(model, x: torch.Tensor) -> list:
    """Run one forward pass, return post-activation maps per conv layer as
    (C, H, W) numpy arrays (5 layers for PilotNet)."""
    acts, handles = _hook_convs(model)
    with torch.no_grad():
        model(x[None] if x.dim() == 3 else x)
    for h in handles:
        h.remove()
    return [a.numpy() for a in acts[:5]]


def gradient_saliency(model, x: torch.Tensor) -> np.ndarray:
    """|d steer / d pixel| averaged over channels, normalised to [0, 1].

    x: (3, H, W) YUV float tensor in [0, 255] scale (the model input).
    """
    xin = x[None].clone().requires_grad_(True)
    model(xin).backward()
    sal = xin.grad[0].abs().mean(0).numpy()
    hi = np.percentile(sal, 99)
    return np.clip(sal / (hi + 1e-9), 0, 1)


def overlay_saliency(rgb_small: np.ndarray, sal: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """Blend the saliency heat (custom navy->teal->amber LUT) into an RGB image.

    rgb_small: (H, W, 3) uint8 — the model-size view of the frame (66x200).
    """
    lut = saliency_lut().astype(np.float64)
    heat = lut[np.clip((sal * 255).astype(np.uint8), 0, 255)]
    a = (alpha * sal)[..., None]
    base = rgb_small.astype(np.float64)
    over = a * heat + (1 - a) * base
    return np.clip(over, 0, 255).astype(np.uint8)


def prep_image(
    path_or_rgb,
):
    """RGB uint8 frame -> (yuv tensor for the model, rgb at model size)."""
    if isinstance(path_or_rgb, str):
        rgb = cv2.cvtColor(cv2.imread(path_or_rgb), cv2.COLOR_BGR2RGB)
    else:
        rgb = path_or_rgb
    yuv = preprocess(rgb)
    x = torch.from_numpy(to_tensor_array(yuv))
    small = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return x, small


# ---------------------------------------------------------------- commands --


def cmd_hist(a):
    import matplotlib.pyplot as plt

    navy_style()
    df = load_log(a.data)
    bal = balance(df)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    ax[0].hist(df["steering"], bins=CFG.n_hist_bins, color=BLUE)
    ax[0].set_title(f"raw steering  (n={len(df)})")
    ax[1].hist(bal["steering"], bins=CFG.n_hist_bins, color=TEAL)
    ax[1].set_title(f"balanced  (n={len(bal)})")
    for axi in ax:
        axi.set_xlabel("steering [-1, 1]")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("saved", a.out)


def cmd_maps(a):
    """Per-layer grid: input | mean map | top-K channels, one row per conv."""
    import matplotlib.pyplot as plt

    navy_style()
    model = load_model_for(a.model)
    x, small = prep_image(a.image)
    acts = activations(model, x)

    K = a.channels
    fig, rows = plt.subplots(len(acts), K + 2, figsize=(1.55 * (K + 2), 1.75 * len(acts)))
    for r, (act, role) in enumerate(zip(acts, CONV_ROLES)):
        top = np.argsort(-act.mean(axis=(1, 2)))[:K]
        cells = [small] + [act.mean(0)] + [act[c] for c in top]
        for c, img in enumerate(cells):
            axr = rows[r, c]
            axr.imshow(img, cmap="viridis" if c > 0 else None)
            axr.set_xticks([]), axr.set_yticks([])
            for s in axr.spines.values():
                s.set_visible(False)
        rows[r, 0].set_ylabel(
            f"conv{r + 1}\n{role}", rotation=0, ha="right", va="center", fontsize=8, color=INK
        )
    # column titles once
    for c, t in enumerate(["input", "mean"] + [f"top {i + 1}" for i in range(K)]):
        rows[0, c].set_title(t, fontsize=7, color=DIM)
    fig.suptitle("WHAT THE LAYERS SEE — lines → shapes → one number", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=140)
    plt.close(fig)
    print("saved", a.out)


def cmd_saliency(a):
    import matplotlib.pyplot as plt

    navy_style()
    model = load_model_for(a.model)
    x, small = prep_image(a.image)
    sal = gradient_saliency(model, x)
    over = overlay_saliency(small, sal)
    steer = float(model(x[None]).item())

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.4))
    panels = [
        (small, "model input 66×200 (YUV shown as RGB)"),
        (over, f"saliency overlay — steer {steer * CFG.sim_max_angle_deg:+.1f}°"),
        (sal, "|∂steer/∂pixel|"),
    ]
    for axi, (img, title) in zip(ax, panels):
        axi.imshow(img, cmap="inferno" if img.ndim == 2 else None)
        axi.set_title(title, fontsize=8, color=INK)
        axi.set_xticks([]), axi.set_yticks([])
        for s in axi.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    plt.close(fig)

    focus = sal[sal > 0.5].mean() if (sal > 0.5).any() else 0.0
    print(f"saved {a.out}  |  steer={steer:+.3f}  high-saliency mass={focus:.2f}")
    print(
        "honesty check: the hot region should sit on lane lines / road edges. "
        "If it is elsewhere, the model may be cheating (e.g. horizon position) — "
        "look before you celebrate."
    )


def cmd_eval(a):
    """Kept for CLI compatibility; the implementation lives in src.evaluate."""
    from .data import DrivingDataset  # noqa: F401  (compat import)
    from .evaluate import evaluate, metrics, plot_scatter

    model = load_model_for(a.model)
    df = load_log(a.data).head(a.n)
    samples = expand_cameras(df, a.data)[::3]  # centre cameras only
    ys, ps = evaluate(model, samples, batch_size=a.batch_size)
    m = metrics(ys, ps)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plot_scatter(ys, ps, a.out, title=f"MAE {m['mae_deg']:.2f} deg (n={m['n']})")
    print(
        f"MAE: {m['mae']:.4f}  ({m['mae_deg']:.2f} deg)   RMSE: {m['rmse']:.4f} "
        f"({m['rmse_deg']:.2f} deg)"
    )
    print("saved", a.out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    h = sp.add_parser("hist")
    h.add_argument("--data", required=True)
    h.add_argument("--out", default="runs/hist.png")

    m = sp.add_parser("maps")
    m.add_argument("--model", required=True)
    m.add_argument("--image", required=True)
    m.add_argument("--channels", type=int, default=6, help="top-K channels per layer")
    m.add_argument("--out", default="runs/maps.png")

    s = sp.add_parser("saliency")
    s.add_argument("--model", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--out", default="runs/saliency.png")

    e = sp.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--n", type=int, default=2000)
    e.add_argument("--batch-size", type=int, default=128)
    e.add_argument("--out", default="runs/eval.png")

    a = ap.parse_args()
    os.makedirs("runs", exist_ok=True)
    {"hist": cmd_hist, "maps": cmd_maps, "saliency": cmd_saliency, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
