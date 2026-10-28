"""Run stage 1 over the whole dataset once and cache the results as .npy files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from visionlingo.data.dataset import list_frames
from .mouth import Detector
from .pipeline import crop_clip


def cache_path(out_dir: str | Path, row) -> Path:
    return Path(out_dir) / f"{row['speaker']}__{row['word']}__{row['instance']}.npy"


def load_cached(path: str | Path) -> np.ndarray:
    return np.load(path)


def process_dataset(df: pd.DataFrame, out_dir: str | Path, img_size: int = 64,
                    detector: Detector | None = None, probe_frames: int = 5,
                    width_scale: float = 0.6, center_y: float = 0.78,
                    fallback_kwargs: dict | None = None, frame_prefix: str = "color_",
                    overwrite: bool = False, log_every: int = 100) -> pd.DataFrame:
    """
    Crop every clip and save it. One bad clip never stops the run: its error is recorded.
    Resumable: existing cache files are reused unless overwrite=True.

    Returns the index with extra columns: cache_path, detected, error.
    (Clips skipped because the cache file already existed have detected = NaN.)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for n, (_, row) in enumerate(df.iterrows(), start=1):
        target = cache_path(out_dir, row)
        rec = {"cache_path": str(target), "detected": np.nan, "error": ""}
        if target.exists() and not overwrite:
            records.append(rec)
        else:
            try:
                clip, info = crop_clip(list_frames(row["path"], frame_prefix), img_size, detector,
                                       probe_frames, width_scale, center_y, fallback_kwargs)
                np.save(target, clip)
                rec["detected"] = info["detected"]
            except Exception as e:                       # keep going, report at the end
                rec["error"] = f"{type(e).__name__}: {e}"
            records.append(rec)
        if log_every and n % log_every == 0:
            print(f"  processed {n}/{len(df)} clips")
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    out.to_csv(out_dir / "processed_index.csv", index=False)
    return out
