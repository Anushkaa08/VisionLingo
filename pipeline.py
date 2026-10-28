"""Stage 1 (slow, run once): raw frames -> small cropped grayscale sequence."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .mouth import Detector, estimate_mouth_box


def crop_clip(frame_paths: Sequence[str | Path], img_size: int = 64, detector: Detector | None = None,
              probe_frames: int = 5, width_scale: float = 0.6, center_y: float = 0.78,
              fallback_kwargs: dict | None = None) -> tuple[np.ndarray, dict]:
    """
    Returns (clip, info):
      clip : uint8 array (n_frames, img_size, img_size), grayscale mouth crops (ALL frames kept,
             so training can later sample/jitter frames without re-reading 640x480 images)
      info : {"detected": bool, "box": (x0,y0,x1,y1), "n_frames": int, "orig_size": (w,h)}

    Why grayscale? Lip *shape and motion* carry the information; colour adds parameters
    but little signal for a small dataset. Why INTER_AREA? Best interpolation for shrinking.
    """
    if len(frame_paths) == 0:
        raise ValueError("Clip has no frames")

    gray_frames = []
    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:                                   # corrupt / unreadable file
            raise ValueError(f"Unreadable frame: {p}")
        gray_frames.append(img)
    shapes = {g.shape for g in gray_frames}
    if len(shapes) != 1:
        raise ValueError(f"Frames in one clip have different sizes: {shapes}")

    box = estimate_mouth_box(gray_frames, detector, probe_frames, width_scale, center_y, fallback_kwargs)
    crops = [cv2.resize(box.crop(g), (img_size, img_size), interpolation=cv2.INTER_AREA)
             for g in gray_frames]
    h, w = gray_frames[0].shape
    info = {"detected": box.detected, "box": box.as_tuple(), "n_frames": len(crops), "orig_size": (w, h)}
    return np.stack(crops).astype(np.uint8), info
