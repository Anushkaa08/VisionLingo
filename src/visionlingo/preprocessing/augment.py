"""
Normalisation and augmentation, on clips of shape (T, H, W).

Augmentation = random, label-preserving changes that fake extra training data so the model
cannot memorise exact pixels. IMPORTANT: the SAME random change is applied to every frame of
a clip, otherwise we would invent motion that never happened.

We deliberately do NOT flip horizontally: it is unnecessary for a small mouth crop and could
blur the learned link between lip shape and sound.
"""
from __future__ import annotations

import numpy as np


def to_float01(clip_uint8: np.ndarray) -> np.ndarray:
    return clip_uint8.astype(np.float32) / 255.0


def normalize_clip(clip01: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-clip standardisation (mean 0, std 1). Removes lighting/contrast differences between
    speakers and recording sessions, so the model focuses on lip shape. A constant (blank) clip
    becomes all zeros instead of dividing by zero."""
    mean, std = clip01.mean(), clip01.std()
    return ((clip01 - mean) / (std + eps)).astype(np.float32)


def augment_clip(clip01: np.ndarray, rng: np.random.Generator, brightness: float = 0.15,
                 contrast: float = 0.15, max_shift: int = 4, noise_std: float = 0.01) -> np.ndarray:
    """clip01: float32 (T,H,W) in [0,1]. Returns float32 in [0,1]."""
    x = clip01.astype(np.float32).copy()

    # contrast around the clip mean, then brightness offset
    c = 1.0 + rng.uniform(-contrast, contrast)
    b = rng.uniform(-brightness, brightness)
    m = x.mean()
    x = (x - m) * c + m + b

    # small translation (mimics imperfect mouth centring); edge padding avoids black borders
    if max_shift > 0:
        dy, dx = (int(v) for v in rng.integers(-max_shift, max_shift + 1, size=2))
        T, H, W = x.shape
        padded = np.pad(x, ((0, 0), (max_shift, max_shift), (max_shift, max_shift)), mode="edge")
        y0, x0 = max_shift + dy, max_shift + dx
        x = padded[:, y0:y0 + H, x0:x0 + W]

    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)

    return np.clip(x, 0.0, 1.0).astype(np.float32)
