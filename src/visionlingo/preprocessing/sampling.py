"""
Stage 2 (cheap, every epoch): pick exactly `seq_len` frames from a clip of any length.

Clips have different lengths, but neural networks need fixed-size batches, so we resample.
  * uniform_indices  -> deterministic, evenly spaced (used for validation/test/inference)
  * jittered_indices -> one random frame per equal-length segment (training augmentation:
                        slightly different timing every epoch, like natural speaking-speed changes)
Short clips simply repeat frames; long clips skip frames.
"""
from __future__ import annotations

import numpy as np


def uniform_indices(n_frames: int, seq_len: int) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    return np.round(np.linspace(0, n_frames - 1, seq_len)).astype(int)


def jittered_indices(n_frames: int, seq_len: int, rng: np.random.Generator) -> np.ndarray:
    if n_frames <= 0 or seq_len <= 0:
        raise ValueError("n_frames and seq_len must be positive")
    edges = np.linspace(0, n_frames, seq_len + 1)
    lo = np.floor(edges[:-1]).astype(int)
    hi = np.minimum(np.maximum(np.ceil(edges[1:]).astype(int), lo + 1), n_frames)  # exclusive
    return np.sort(rng.integers(lo, hi))


def sample_clip(clip: np.ndarray, seq_len: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """clip: (n_frames, H, W). rng=None -> deterministic uniform sampling."""
    n = clip.shape[0]
    idx = uniform_indices(n, seq_len) if rng is None else jittered_indices(n, seq_len, rng)
    return clip[idx]
