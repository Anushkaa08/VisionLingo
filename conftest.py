import cv2
import numpy as np
import pytest


@pytest.fixture
def fake_dataset(tmp_path):
    """Tiny synthetic dataset with the same folder layout as MIRACL-VC1.
    5 speakers x 3 words x 2 instances; instance lengths vary like real clips."""
    root = tmp_path / "MIRACL-VC1"
    rng = np.random.default_rng(0)
    for spk in ["F01", "F02", "F03", "M01", "M02"]:
        for word in ["01", "02", "03"]:
            for inst in ["01", "02"]:
                d = root / spk / "words" / word / inst
                d.mkdir(parents=True)
                for i in range(1, int(rng.integers(4, 9)) + 1):
                    cv2.imwrite(str(d / f"color_{i:03d}.jpg"), rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
                    cv2.imwrite(str(d / f"depth_{i:03d}.jpg"), rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
    return root
