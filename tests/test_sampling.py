import numpy as np
import pytest

from visionlingo.preprocessing import jittered_indices, sample_clip, uniform_indices


def test_uniform_length_and_endpoints():
    idx = uniform_indices(40, 24)
    assert len(idx) == 24 and idx[0] == 0 and idx[-1] == 39
    assert (np.diff(idx) >= 0).all()


def test_short_clip_repeats_frames_but_keeps_length():
    idx = uniform_indices(3, 8)
    assert len(idx) == 8 and idx.min() == 0 and idx.max() == 2


def test_single_frame_clip():
    assert set(uniform_indices(1, 5)) == {0}


def test_jittered_is_sorted_in_range_and_correct_length():
    rng = np.random.default_rng(0)
    for n in [1, 5, 24, 37, 100]:
        idx = jittered_indices(n, 24, rng)
        assert len(idx) == 24 and idx.min() >= 0 and idx.max() <= n - 1
        assert (np.diff(idx) >= 0).all()


def test_jitter_is_reproducible_and_actually_varies():
    a = jittered_indices(60, 24, np.random.default_rng(1))
    b = jittered_indices(60, 24, np.random.default_rng(1))
    c = jittered_indices(60, 24, np.random.default_rng(2))
    assert (a == b).all() and not (a == c).all()


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        uniform_indices(0, 24)
    with pytest.raises(ValueError):
        uniform_indices(10, 0)


def test_sample_clip_shape():
    clip = np.zeros((31, 8, 8), np.uint8)
    assert sample_clip(clip, 24).shape == (24, 8, 8)
    assert sample_clip(clip, 24, np.random.default_rng(0)).shape == (24, 8, 8)
