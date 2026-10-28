"""
Train / validation / test splits.

WHY SPEAKER-INDEPENDENT? If the same person appears in train and test, the model can learn
to recognise *faces* rather than *lip movements* (data leakage), inflating accuracy.
`speaker_split` holds out whole speakers. `random_split` exists only so we can show the
inflated number next to the honest one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def speaker_split(df: pd.DataFrame, n_val: int = 2, n_test: int = 2, seed: int = 42) -> dict[str, pd.DataFrame]:
    speakers = np.array(sorted(df["speaker"].unique()))
    if n_val < 1 or n_test < 1:
        raise ValueError("n_val and n_test must each be at least 1")
    if n_val + n_test >= len(speakers):
        raise ValueError(f"Need more than {n_val + n_test} speakers for training; found {len(speakers)}")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(speakers)
    test_sp = set(shuffled[:n_test])
    val_sp = set(shuffled[n_test:n_test + n_val])
    splits = {
        "train": df[~df["speaker"].isin(test_sp | val_sp)],
        "val": df[df["speaker"].isin(val_sp)],
        "test": df[df["speaker"].isin(test_sp)],
    }
    assert_no_speaker_overlap(splits)
    return {k: v.reset_index(drop=True) for k, v in splits.items()}


def random_split(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Stratified 70/15/15 random split. Speakers LEAK across sets: for comparison only."""
    train, tmp = train_test_split(df, test_size=0.3, stratify=df["word"], random_state=seed)
    val, test = train_test_split(tmp, test_size=0.5, stratify=tmp["word"], random_state=seed)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def assert_no_speaker_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """Fail loudly if any speaker appears in more than one split."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a]["speaker"]) & set(splits[b]["speaker"])
            if shared:
                raise AssertionError(f"Speaker leakage between {a} and {b}: {sorted(shared)}")
