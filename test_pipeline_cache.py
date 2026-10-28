import numpy as np
import pytest

from visionlingo.data import build_index
from visionlingo.data.dataset import list_frames
from visionlingo.preprocessing import crop_clip, load_cached, process_dataset

FAKE_FACE = lambda gray: [(2, 2, 12, 12)]       # pretend a face was found (images are 16x16)
NO_FACE = lambda gray: []


def test_crop_clip_shape_dtype_and_info(fake_dataset):
    row = build_index(fake_dataset).iloc[0]
    clip, info = crop_clip(list_frames(row.path), img_size=32, detector=FAKE_FACE)
    assert clip.shape == (row.n_frames, 32, 32) and clip.dtype == np.uint8
    assert info["detected"] is True and info["n_frames"] == row.n_frames


def test_crop_clip_flags_fallback(fake_dataset):
    row = build_index(fake_dataset).iloc[0]
    _, info = crop_clip(list_frames(row.path), detector=NO_FACE)
    assert info["detected"] is False


def test_unreadable_frame_raises(fake_dataset):
    row = build_index(fake_dataset).iloc[0]
    frames = list_frames(row.path)
    frames[0].write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unreadable frame"):
        crop_clip(frames, detector=FAKE_FACE)


def test_process_dataset_roundtrip_and_resume(fake_dataset, tmp_path):
    df = build_index(fake_dataset)
    out = process_dataset(df, tmp_path / "cache", 32, FAKE_FACE, log_every=0)
    assert (out["error"] == "").all() and out["detected"].all()
    first = out.iloc[0]
    arr = load_cached(first["cache_path"])
    assert arr.shape == (first["n_frames"], 32, 32)
    assert (tmp_path / "cache" / "processed_index.csv").exists()

    again = process_dataset(df, tmp_path / "cache", 32, FAKE_FACE, log_every=0)   # resume
    assert again["detected"].isna().all()                                           # nothing recomputed


def test_bad_clip_is_recorded_not_fatal(fake_dataset, tmp_path):
    df = build_index(fake_dataset)
    list_frames(df.iloc[3].path)[0].write_bytes(b"garbage")
    out = process_dataset(df, tmp_path / "c", 32, FAKE_FACE, log_every=0)
    assert (out["error"] != "").sum() == 1 and (out["error"] == "").sum() == len(df) - 1
