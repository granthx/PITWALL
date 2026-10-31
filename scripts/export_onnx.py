"""Export a trained steering model to ONNX and verify parity with PyTorch.

    python scripts/export_onnx.py --model models/pilotnet.pt
    python scripts/export_onnx.py --model models/pilotnet.pt --data data/synthetic

Writes models/<name>.onnx. If ``onnxruntime`` is installed, runs a parity
check on random inputs (and on real preprocessed frames when --data is given)
and prints the max |PyTorch - ONNX| difference. Steering parity should be
tight (< 1e-4 for fp32).
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.model import build_model
from src.preprocess import preprocess, to_tensor_array


def parity(model, onnx_path, frames):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping parity check " "(pip install onnx onnxruntime)")
        return
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.stack(frames).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out = sess.run(None, {"frame": x})[0]
    diff = np.abs(ref - out)
    print(
        f"parity on {len(x)} frames: max|torch-onnx| = {diff.max():.2e}  "
        f"mean = {diff.mean():.2e}"
    )
    if diff.max() > 1e-3:
        print("WARNING: divergence above 1e-3 — inspect before deploying")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pilotnet.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=None, help="data dir for real-frame parity check")
    ap.add_argument("--n", type=int, default=8, help="frames for the parity check")
    a = ap.parse_args()

    ckpt = torch.load(a.model, map_location="cpu")
    arch = ckpt.get("arch", "pilotnet")
    model = build_model(arch)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out = a.out or os.path.splitext(a.model)[0] + f"_{arch}.onnx"
    dummy = torch.zeros(1, CFG.img_c, CFG.img_h, CFG.img_w)
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["frame"],
        output_names=["steer"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={"frame": {0: "batch"}, "steer": {0: "batch"}},
    )
    print(f"exported {out} (arch={arch}, opset 13, dynamic batch)")

    rng = np.random.default_rng(0)
    frames = [
        rng.integers(0, 255, (CFG.img_c, CFG.img_h, CFG.img_w)).astype(np.float32)
        for _ in range(a.n)
    ]
    if a.data:
        import glob

        import cv2

        paths = sorted(glob.glob(os.path.join(a.data, "IMG", "center_*.jpg")))[: a.n]
        for p in paths:
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            frames.append(to_tensor_array(preprocess(rgb)))
    parity(model, out, frames)


if __name__ == "__main__":
    main()
