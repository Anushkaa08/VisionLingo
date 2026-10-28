"""
Index the MIRACL-VC1 dataset.

Assumed on-disk layout (verify against your download with scripts/explore_dataset.py):

    <root>/<speaker>/<subset>/<word_id>/<instance>/color_001.jpg, color_002.jpg, ...

Each *instance* folder is ONE spoken word = one training example (a sequence of frames).
We build a table with one row per instance. No images are loaded here, only paths and counts,
so indexing 1,500 clips takes about a second.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_NUM = re.compile(r"(\d+)")


def _natural_key(path: Path) -> list:
    """Sort color_2.jpg before color_10.jpg even if the names aren't zero-padded."""
    return [int(t) if t.isdigit() else t for t in _NUM.split(path.name)]


def list_frames(instance_dir: str | Path, prefix: str = "color_") -> list[Path]:
    """Frame files of one clip, in temporal order."""
    files = [p for p in Path(instance_dir).iterdir()
             if p.is_file() and p.name.startswith(prefix)
             and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return sorted(files, key=_natural_key)


def build_index(root: str | Path, subset: str = "words", prefix: str = "color_") -> pd.DataFrame:
    """Return a DataFrame: speaker, word, instance, n_frames, path (one row per clip)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {root.resolve()}\n"
            "Download MIRACL-VC1 and extract it there (see data/README.md).")

    rows = []
    for speaker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subset_dir = speaker_dir / subset
        if not subset_dir.is_dir():
            continue                                  # skip stray folders
        for word_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
            for inst_dir in sorted(p for p in word_dir.iterdir() if p.is_dir()):
                n = len(list_frames(inst_dir, prefix))
                if n == 0:
                    continue                          # empty folder, not a usable clip
                rows.append({"speaker": speaker_dir.name, "word": word_dir.name,
                             "instance": inst_dir.name, "n_frames": n,
                             "path": str(inst_dir)})

    if not rows:
        raise ValueError(
            f"No clips found under {root.resolve()} using the pattern "
            f"<speaker>/{subset}/<word>/<instance>/{prefix}*.jpg. "
            "Check the folder layout and adjust configs/config.yaml.")
    return pd.DataFrame(rows)
