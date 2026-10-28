"""
Phase 1 sanity check: does the loader see your dataset correctly?

Run from the project root:   python scripts/explore_dataset.py
Prints dataset statistics and saves two figures into docs/.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionlingo.data import build_index, speaker_split        # noqa: E402
from visionlingo.data.dataset import list_frames                # noqa: E402


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    d, s = cfg["data"], cfg["split"]
    df = build_index(ROOT / d["root"], d["subset"], d["frame_prefix"])

    print(f"Clips: {len(df)} | speakers: {df.speaker.nunique()} | word classes: {df.word.nunique()}")
    print("\nClips per word:\n", df.groupby("word").size().to_string())
    print("\nClips per speaker:\n", df.groupby("speaker").size().to_string())
    print("\nFrames per clip:\n", df.n_frames.describe().round(1).to_string())

    splits = speaker_split(df, s["n_val_speakers"], s["n_test_speakers"], s["seed"])
    print("\nSpeaker-independent split (no speaker overlap verified):")
    for name, part in splits.items():
        print(f"  {name:5s}: {len(part):4d} clips | speakers: {sorted(part.speaker.unique())}")

    (ROOT / "docs").mkdir(exist_ok=True)

    # Figure 1: clip-length distribution (why we must resample to a fixed number of frames)
    plt.figure(figsize=(6, 4))
    plt.hist(df.n_frames, bins=30, color="#4C78A8")
    plt.xlabel("Frames per clip"); plt.ylabel("Number of clips"); plt.title("Clip length distribution")
    plt.tight_layout(); plt.savefig(ROOT / "docs" / "frames_per_clip.png", dpi=130); plt.close()

    # Figure 2: middle frame of one clip per word
    words = sorted(df.word.unique())
    fig, axes = plt.subplots(2, (len(words) + 1) // 2, figsize=(14, 6))
    for ax, w in zip(axes.ravel(), words):
        row = df[df.word == w].iloc[0]
        frames = list_frames(row.path, d["frame_prefix"])
        img = cv2.cvtColor(cv2.imread(str(frames[len(frames) // 2])), cv2.COLOR_BGR2RGB)
        ax.imshow(img); ax.set_title(f"word {w} ({row.speaker})"); ax.axis("off")
    plt.tight_layout(); plt.savefig(ROOT / "docs" / "sample_frames.png", dpi=110); plt.close()
    print("\nSaved docs/frames_per_clip.png and docs/sample_frames.png")


if __name__ == "__main__":
    main()
