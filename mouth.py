"""
Locate the mouth region.

Strategy (classical computer vision, no extra model downloads):
  1. Detect the face with OpenCV's Haar cascade on a few frames of the clip.
  2. Take the MEDIAN face box -> one stable box for the whole clip (no frame-to-frame jitter,
     robust to a single bad detection).
  3. Derive a square mouth box from the face box using fixed proportions
     (a face is roughly: eyes in the upper half, mouth around 75-80% of the way down).
  4. If no face is found, use a fixed fallback box and FLAG it so we can report how often it happened.

Limitations (documented honestly): one box per clip assumes the head barely moves; Haar cascades
fail on strong head turns or poor lighting; proportions are approximate. Landmark models
(dlib / MediaPipe) are more precise, which is listed as a future improvement.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

import cv2
import numpy as np

Face = tuple[int, int, int, int]                 # x, y, w, h
Detector = Callable[[np.ndarray], Sequence[Face]]  # gray image -> list of faces


@dataclass(frozen=True)
class MouthBox:
    x0: int
    y0: int
    x1: int
    y1: int
    detected: bool          # False => fallback box was used

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y0:self.y1, self.x0:self.x1]

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


@lru_cache(maxsize=1)
def default_detector(scale_factor: float = 1.1, min_neighbors: int = 5, min_size: int = 80) -> Detector:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Could not load the Haar cascade bundled with OpenCV.")

    def detect(gray: np.ndarray) -> Sequence[Face]:
        faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor,
                                         minNeighbors=min_neighbors, minSize=(min_size, min_size))
        return [tuple(int(v) for v in f) for f in faces]
    return detect


def _square_box(cx: float, cy: float, side: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Square of size `side` centred at (cx, cy), shifted (not shrunk) to stay inside the image."""
    side = max(2, min(side, img_w, img_h))
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = min(max(x0, 0), img_w - side)
    y0 = min(max(y0, 0), img_h - side)
    return x0, y0, x0 + side, y0 + side


def mouth_box_from_face(face: Face, img_w: int, img_h: int,
                        width_scale: float = 0.6, center_y: float = 0.78) -> tuple[int, int, int, int]:
    x, y, w, h = face
    return _square_box(x + w / 2, y + center_y * h, int(round(width_scale * w)), img_w, img_h)


def fallback_box(img_w: int, img_h: int, side_frac: float = 0.35,
                 center_x: float = 0.5, center_y: float = 0.62) -> MouthBox:
    x0, y0, x1, y1 = _square_box(center_x * img_w, center_y * img_h, int(side_frac * img_w), img_w, img_h)
    return MouthBox(x0, y0, x1, y1, detected=False)


def estimate_mouth_box(gray_frames: Sequence[np.ndarray], detector: Detector | None = None,
                       probe_frames: int = 5, width_scale: float = 0.6, center_y: float = 0.78,
                       fallback_kwargs: dict | None = None) -> MouthBox:
    """One stable mouth box for a whole clip."""
    if len(gray_frames) == 0:
        raise ValueError("No frames given")
    detector = detector or default_detector()
    h, w = gray_frames[0].shape[:2]

    probe_idx = np.unique(np.linspace(0, len(gray_frames) - 1, probe_frames).round().astype(int))
    faces = []
    for i in probe_idx:
        found = detector(gray_frames[i])
        if len(found):
            faces.append(max(found, key=lambda f: f[2] * f[3]))   # largest face = the speaker

    if not faces:
        return fallback_box(w, h, **(fallback_kwargs or {}))

    median_face = tuple(int(v) for v in np.median(np.array(faces), axis=0))
    x0, y0, x1, y1 = mouth_box_from_face(median_face, w, h, width_scale, center_y)
    return MouthBox(x0, y0, x1, y1, detected=True)
