"""Image preprocessing shared by training AND driving.

Training and inference MUST use the exact same function, otherwise the car
sees a different world than the one it learned in (the #1 cause of a car that
trains well but drives badly).
"""

import cv2
import numpy as np

from .config import CFG


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (160x320x3) -> YUV uint8 (66x200x3), cropped and resized."""
    img = img_rgb[CFG.crop_top : img_rgb.shape[0] - CFG.crop_bottom, :, :]
    img = cv2.resize(img, (CFG.img_w, CFG.img_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def to_tensor_array(img_yuv: np.ndarray) -> np.ndarray:
    """HWC uint8 -> CHW float32 in [0, 255]. Scaling to [-1, 1] happens inside the model."""
    return np.ascontiguousarray(img_yuv.transpose(2, 0, 1), dtype=np.float32)
