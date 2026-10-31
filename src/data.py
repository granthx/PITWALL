"""Loading the Udacity driving_log.csv, balancing it, and serving batches."""
import os
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import augment
from .config import CFG
from .preprocess import preprocess, to_tensor_array

COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed"]


def _resolve(data_dir: str, p: str) -> str:
    """The CSV stores absolute paths from the recording machine (often Windows).
    Keep only the file name and look in <data_dir>/IMG/."""
    name = str(p).strip().replace("\\", "/").split("/")[-1]
    return os.path.join(data_dir, "IMG", name)


def load_log(data_dir: str) -> pd.DataFrame:
    csv = os.path.join(data_dir, "driving_log.csv")
    df = pd.read_csv(csv, header=None, names=COLUMNS)
    # Tolerate a header row if present
    df = df[pd.to_numeric(df["steering"], errors="coerce").notna()].copy()
    df["steering"] = df["steering"].astype(float)
    return df.reset_index(drop=True)


def balance(df: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Most driving is straight. Without balancing the model learns 'always go
    straight'. Drop most near-zero samples, then cap each steering-histogram bin."""
    rng = np.random.default_rng(cfg.seed)
    straight = df["steering"].abs() < cfg.zero_threshold
    keep = ~straight | (rng.random(len(df)) < cfg.zero_keep_prob)
    df = df[keep].reset_index(drop=True)

    bins = np.linspace(-1, 1, cfg.n_hist_bins + 1)
    idx = np.digitize(df["steering"].values, bins)
    drop: List[int] = []
    for b in np.unique(idx):
        members = np.where(idx == b)[0]
        if len(members) > cfg.max_per_bin:
            drop.extend(rng.choice(members, len(members) - cfg.max_per_bin, replace=False))
    return df.drop(index=drop).reset_index(drop=True)


def expand_cameras(df: pd.DataFrame, data_dir: str, cfg=CFG) -> List[Tuple[str, float]]:
    """Turn each row into 3 samples (center/left/right) with steering offsets.
    Left/right cameras teach the car how to recover back to the lane centre."""
    samples = []
    for _, r in df.iterrows():
        a = float(r["steering"])
        samples.append((_resolve(data_dir, r["center"]), a))
        samples.append((_resolve(data_dir, r["left"]), a + cfg.steering_correction))
        samples.append((_resolve(data_dir, r["right"]), a - cfg.steering_correction))
    return [(p, float(np.clip(a, -1, 1))) for p, a in samples]


class DrivingDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, float]], training: bool, seed: int = 0):
        self.samples = samples
        self.training = training
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, angle = self.samples[i]
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.training:
            img, angle = augment(img, angle, self.rng)
        x = torch.from_numpy(to_tensor_array(preprocess(img)))
        return x, torch.tensor([angle], dtype=torch.float32)
