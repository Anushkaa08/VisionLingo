import numpy as np
import pytest

from visionlingo.preprocessing import estimate_mouth_box, fallback_box, mouth_box_from_face


def test_mouth_box_is_square_and_inside_image():
    x0, y0, x1, y1 = mouth_box_from_face((200, 100, 200, 240), 640, 480)
    assert x1 - x0 == y1 - y0
    assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480


def test_mouth_box_is_in_lower_half_of_face():
    face = (200, 100, 200, 240)
    _, y0, _, y1 = mouth_box_from_face(face, 640, 480)
    assert (y0 + y1) / 2 > face[1] + 0.5 * face[3]


def test_box_shifted_not_shrunk_at_image_border():
    x0, y0, x1, y1 = mouth_box_from_face((560, 300, 200, 240), 640, 480)  # face hangs off the edge
    assert x1 <= 640 and y1 <= 480 and (x1 - x0) == (y1 - y0) >= 100


def test_fallback_is_flagged_and_inside_image():
    b = fallback_box(640, 480)
    assert b.detected is False and 0 <= b.x0 < b.x1 <= 640 and 0 <= b.y0 < b.y1 <= 480


def test_estimate_uses_detector_and_median():
    faces = iter([[(100, 100, 200, 200)], [(110, 100, 200, 200)], [(120, 100, 200, 200)],
                  [(5000, 5000, 1, 1)], [(100, 100, 200, 200)]])
    frames = [np.zeros((480, 640), np.uint8)] * 5
    box = estimate_mouth_box(frames, detector=lambda g: next(faces), probe_frames=5)
    assert box.detected is True


def test_estimate_picks_largest_face():
    big, small = (100, 100, 200, 200), (400, 50, 50, 50)
    frames = [np.zeros((480, 640), np.uint8)] * 3
    box = estimate_mouth_box(frames, detector=lambda g: [small, big], probe_frames=3)
    assert box.detected and 100 <= box.x0 <= 300          # box sits on the big face, not the small one


def test_no_face_uses_fallback():
    frames = [np.zeros((120, 160), np.uint8)] * 4
    assert estimate_mouth_box(frames, detector=lambda g: []).detected is False


def test_empty_clip_raises():
    with pytest.raises(ValueError):
        estimate_mouth_box([], detector=lambda g: [])
