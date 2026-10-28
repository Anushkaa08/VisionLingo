import pytest

from visionlingo.data import assert_no_speaker_overlap, build_index, random_split, speaker_split


def test_speaker_split_has_no_leakage(fake_dataset):
    df = build_index(fake_dataset)
    s = speaker_split(df, n_val=1, n_test=1, seed=1)
    assert_no_speaker_overlap(s)
    assert sum(len(v) for v in s.values()) == len(df)          # nothing lost or duplicated
    assert s["train"].speaker.nunique() == 3


def test_speaker_split_is_reproducible(fake_dataset):
    df = build_index(fake_dataset)
    a, b = speaker_split(df, 1, 1, seed=7), speaker_split(df, 1, 1, seed=7)
    assert list(a["test"].speaker.unique()) == list(b["test"].speaker.unique())


def test_too_few_speakers_raises(fake_dataset):
    with pytest.raises(ValueError):
        speaker_split(build_index(fake_dataset), n_val=3, n_test=2)


def test_random_split_DOES_leak_speakers(fake_dataset):
    """Documents WHY we don't use random splits: speakers appear in several sets."""
    s = random_split(build_index(fake_dataset), seed=0)
    with pytest.raises(AssertionError):
        assert_no_speaker_overlap(s)


def test_leakage_detector_catches_overlap(fake_dataset):
    df = build_index(fake_dataset)
    with pytest.raises(AssertionError, match="Speaker leakage"):
        assert_no_speaker_overlap({"train": df, "test": df})
