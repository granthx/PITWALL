"""Data augmentation. All functions take/return RGB uint8 images in raw
simulator size (160x320) plus a steering angle in [-1, 1]."""

import cv2
import numpy as np


def random_flip(img, angle, rng):
    if rng.random() < 0.5:
        return cv2.flip(img, 1), -angle
    return img, angle


def random_translate(img, angle, rng, range_x=60, range_y=20, per_px=0.004):
    """Shift the image; compensate steering proportionally to horizontal shift."""
    tx = range_x * (rng.random() - 0.5)
    ty = range_y * (rng.random() - 0.5)
    angle = angle + tx * per_px
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE), angle


def random_brightness(img, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] *= 0.4 + 1.2 * rng.random()
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_shadow(img, rng):
    h, w = img.shape[:2]
    x1, x2 = w * rng.random(), w * rng.random()
    xm, ym = np.meshgrid(np.arange(w), np.arange(h))
    mask = (ym * (x2 - x1) - h * (xm - x1)) > 0
    ratio = 0.2 + 0.5 * rng.random()
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
    hls[:, :, 1][mask] *= ratio
    hls[:, :, 1] = np.clip(hls[:, :, 1], 0, 255)
    return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2RGB)


def augment(img, angle, rng):
    img, angle = random_flip(img, angle, rng)
    if rng.random() < 0.7:
        img, angle = random_translate(img, angle, rng)
    if rng.random() < 0.5:
        img = random_shadow(img, rng)
    if rng.random() < 0.7:
        img = random_brightness(img, rng)
    return img, float(np.clip(angle, -1.0, 1.0))
