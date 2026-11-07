import os
import subprocess
import sys

import numpy as np
import pytest

from src.augment import augment
from src.config import CFG
from src.preprocess import preprocess, to_tensor_array

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_preprocess_shape():
    out = preprocess(np.zeros((160, 320, 3), np.uint8))
    assert out.shape == (CFG.img_h, CFG.img_w, 3)
    assert to_tensor_array(out).shape == (3, CFG.img_h, CFG.img_w)


def test_augment_bounds():
    rng = np.random.default_rng(1)
    img = np.random.default_rng(2).integers(0, 255, (160, 320, 3), dtype=np.uint8)
    for _ in range(50):
        a, ang = augment(img, 0.3, rng)
        assert a.shape == img.shape and -1 <= ang <= 1


def test_flip_negates_angle():
    from src.augment import random_flip

    class R:  # force a flip
        def random(self):
            return 0.0

    _, ang = random_flip(np.zeros((4, 4, 3), np.uint8), 0.5, R())
    assert ang == -0.5


def test_balance_and_expand(tmp_path):
    pytest.importorskip("torch")
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(ROOT, "scripts/make_synthetic_data.py"),
            "--out",
            str(tmp_path),
            "--n",
            "40",
        ]
    )
    from src.data import DrivingDataset, balance, expand_cameras, load_log

    df = load_log(str(tmp_path))
    assert len(df) == 40
    s = expand_cameras(balance(df), str(tmp_path))
    assert len(s) % 3 == 0
    x, _ = DrivingDataset(s, True)[0]
    assert tuple(x.shape) == (3, CFG.img_h, CFG.img_w)


def test_model_forward():
    torch = pytest.importorskip("torch")
    from src.model import build_model

    for arch in ("pilotnet", "resnet18", "spatial_resnet18"):
        channels = 9 if arch == "spatial_resnet18" else 3
        model = build_model(arch, in_channels=channels)
        out = model(torch.zeros(2, channels, CFG.img_h, CFG.img_w))
        assert tuple(out.shape) == (2, 1), arch


def test_temporal_stacking(tmp_path):
    pytest.importorskip("torch")
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(ROOT, "scripts/make_synthetic_data.py"),
            "--out",
            str(tmp_path),
            "--n",
            "10",
        ]
    )
    from src.data import DrivingDataset, expand_cameras, load_log
    from src.model import build_model

    df = load_log(str(tmp_path))
    s = expand_cameras(df, str(tmp_path), frame_stack=3)
    assert len(s) == len(df) * 3
    x, _y = DrivingDataset(s, training=False)[0]
    assert tuple(x.shape) == (9, CFG.img_h, CFG.img_w)

    for arch in ("pilotnet", "resnet18", "spatial_resnet18"):
        model = build_model(arch, in_channels=9)
        out = model(x.unsqueeze(0))
        assert tuple(out.shape) == (1, 1), f"temporal {arch}"
