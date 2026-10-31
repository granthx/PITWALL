"""Generate a tiny FAKE dataset in the Udacity format so you can smoke-test the
whole pipeline without the simulator.  (A road that curves left/right; the
steering label is derived from the curvature.)

    python scripts/make_synthetic_data.py --out data/synthetic --n 300
"""
import argparse
import os

import cv2
import numpy as np


def frame(curve, rng):
    img = np.zeros((160, 320, 3), np.uint8)
    img[:70] = (200, 190, 170)        # sky
    img[70:] = (60, 90, 60)           # grass
    horizon_x = 160 + curve * 120
    pts = np.array([[60, 160], [260, 160], [horizon_x + 15, 70], [horizon_x - 15, 70]], np.int32)
    cv2.fillPoly(img, [pts], (70, 70, 75))
    cv2.line(img, (160, 160), (int(horizon_x), 70), (230, 200, 40), 2)
    noise = rng.integers(-8, 8, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(a.out, "IMG"), exist_ok=True)
    rows = []
    for i in range(a.n):
        curve = float(rng.uniform(-1, 1))
        steer = float(np.clip(curve * 0.8, -1, 1))
        names = []
        for cam, shift in (("center", 0), ("left", -10), ("right", 10)):
            f = np.roll(frame(curve, rng), shift, axis=1)
            n = f"{cam}_{i:05d}.jpg"
            cv2.imwrite(os.path.join(a.out, "IMG", n), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            names.append(f"IMG/{n}")
        rows.append(",".join(names + [f"{steer:.4f}", "1", "0", "15"]))
    with open(os.path.join(a.out, "driving_log.csv"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"wrote {a.n} rows to {a.out}")


if __name__ == "__main__":
    main()
