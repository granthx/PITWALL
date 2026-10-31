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
    from src.model import PilotNet

    out = PilotNet()(torch.zeros(2, 3, CFG.img_h, CFG.img_w))
    assert tuple(out.shape) == (2, 1)
