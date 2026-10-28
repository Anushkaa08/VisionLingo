import numpy as np

from visionlingo.preprocessing import augment_clip, normalize_clip
from visionlingo.preprocessing.augment import to_float01


def _clip(seed=0):
    return np.random.default_rng(seed).random((24, 16, 16)).astype(np.float32)


def test_augment_keeps_shape_dtype_and_range():
    out = augment_clip(_clip(), np.random.default_rng(1))
    assert out.shape == (24, 16, 16) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_augment_is_reproducible_with_same_seed_and_varies_otherwise():
    a = augment_clip(_clip(), np.random.default_rng(5))
    b = augment_clip(_clip(), np.random.default_rng(5))
    c = augment_clip(_clip(), np.random.default_rng(6))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_augment_does_not_modify_input():
    x = _clip(); before = x.copy()
    augment_clip(x, np.random.default_rng(0))
    assert np.array_equal(x, before)


def test_same_transform_for_all_frames():
    """Identical frames must stay identical in time (no invented motion), apart from tiny noise."""
    frame = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    clip = np.repeat(frame[None], 10, axis=0)
    out = augment_clip(clip, np.random.default_rng(3), noise_std=0.0)
    assert np.allclose(out, out[0:1], atol=1e-6)


def test_no_augmentation_when_all_strengths_zero():
    x = _clip()
    out = augment_clip(x, np.random.default_rng(0), 0, 0, 0, 0)
    assert np.allclose(out, x, atol=1e-6)


def test_normalize_zero_mean_unit_std():
    out = normalize_clip(_clip())
    assert abs(out.mean()) < 1e-4 and abs(out.std() - 1) < 1e-3


def test_normalize_constant_clip_has_no_nan():
    out = normalize_clip(np.full((24, 8, 8), 0.5, np.float32))
    assert np.isfinite(out).all() and np.allclose(out, 0)


def test_to_float01_range():
    out = to_float01(np.array([[[0, 255]]], np.uint8))
    assert out.min() == 0.0 and out.max() == 1.0
