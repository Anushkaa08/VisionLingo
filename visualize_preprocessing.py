"""
ALWAYS look at your data. This saves docs/preprocessing_check.png showing, for a few random
clips: the original middle frame with the detected mouth box, then 8 frames of the cropped
sequence after sampling, and an augmented version of the same sequence.

Run:  python scripts/visualize_preprocessing.py   (after preprocess_dataset.py)
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionlingo.data import build_index                                   # noqa: E402
from visionlingo.data.dataset import list_frames                           # noqa: E402
from visionlingo.preprocessing import augment_clip, crop_clip, sample_clip  # noqa: E402
from visionlingo.preprocessing.augment import to_float01                   # noqa: E402
from visionlingo.preprocessing.mouth import default_detector               # noqa: E402


def main(n_clips: int = 3, seed: int = 0) -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    d, p, a = cfg["data"], cfg["preprocess"], cfg["augment"]
    df = build_index(ROOT / d["root"], d["subset"], d["frame_prefix"])
    rng = np.random.default_rng(seed)
    detector = default_detector(p["face"]["scale_factor"], p["face"]["min_neighbors"], p["face"]["min_size"])
    fb = {"side_frac": p["fallback"]["side_frac"], "center_x": p["fallback"]["center_x"],
          "center_y": p["fallback"]["center_y"]}

    show = 8
    fig, axes = plt.subplots(n_clips * 2, show + 1, figsize=(2 * (show + 1), 4.2 * n_clips))
    for k, i in enumerate(rng.choice(len(df), n_clips, replace=False)):
        row = df.iloc[i]
        frames = list_frames(row.path, d["frame_prefix"])
        clip, info = crop_clip(frames, p["img_size"], detector, p["probe_frames"],
                               p["mouth"]["width_scale"], p["mouth"]["center_y"], fb)
        orig = cv2.cvtColor(cv2.imread(str(frames[len(frames) // 2])), cv2.COLOR_BGR2RGB)
        x0, y0, x1, y1 = info["box"]
        cv2.rectangle(orig, (x0, y0), (x1, y1), (0, 255, 0) if info["detected"] else (255, 0, 0), 3)

        seq = to_float01(sample_clip(clip, p["seq_len"]))
        aug = augment_clip(seq, rng, a["brightness"], a["contrast"], a["max_shift"], a["noise_std"])
        pick = np.linspace(0, p["seq_len"] - 1, show).astype(int)

        for r, (data, label) in enumerate([(seq, "sampled"), (aug, "augmented")]):
            ax = axes[2 * k + r]
            ax[0].imshow(orig) if r == 0 else ax[0].axis("off")
            ax[0].set_title(f"{row.speaker} word {row.word}\n{'face found' if info['detected'] else 'FALLBACK box'}",
                            fontsize=8) if r == 0 else None
            ax[0].axis("off")
            for j, fi in enumerate(pick):
                ax[j + 1].imshow(data[fi], cmap="gray", vmin=0, vmax=1)
                ax[j + 1].set_title(f"{label} f{fi}", fontsize=7); ax[j + 1].axis("off")
    plt.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    plt.savefig(ROOT / "docs" / "preprocessing_check.png", dpi=110)
    print("Saved docs/preprocessing_check.png (green box = face found, red = fallback)")


if __name__ == "__main__":
    main()
