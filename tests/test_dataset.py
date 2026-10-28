import cv2
import numpy as np
import pytest

from visionlingo.data import build_index
from visionlingo.data.dataset import list_frames


def test_index_counts(fake_dataset):
    df = build_index(fake_dataset)
    assert len(df) == 5 * 3 * 2
    assert df.speaker.nunique() == 5 and df.word.nunique() == 3
    assert (df.n_frames >= 4).all()


def test_depth_images_are_ignored(fake_dataset):
    row = build_index(fake_dataset).iloc[0]
    frames = list_frames(row.path)
    assert len(frames) == row.n_frames
    assert all(f.name.startswith("color_") for f in frames)


def test_frames_are_in_temporal_order(tmp_path):
    d = tmp_path / "S" / "words" / "01" / "01"
    d.mkdir(parents=True)
    for i in [1, 2, 10, 11]:                       # not zero-padded on purpose
        cv2.imwrite(str(d / f"color_{i}.jpg"), np.zeros((4, 4, 3), np.uint8))
    assert [f.name for f in list_frames(d)] == ["color_1.jpg", "color_2.jpg", "color_10.jpg", "color_11.jpg"]


def test_missing_root_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset folder not found"):
        build_index(tmp_path / "nope")


def test_wrong_layout_gives_helpful_error(tmp_path):
    (tmp_path / "S1" / "something_else").mkdir(parents=True)
    with pytest.raises(ValueError, match="No clips found"):
        build_index(tmp_path)
