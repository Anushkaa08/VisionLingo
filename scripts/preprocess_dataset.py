"""
Phase 2, stage 1: crop the mouth region of every clip and cache it.

Run from the project root:   python scripts/preprocess_dataset.py
Safe to re-run (it resumes). Use --overwrite to redo everything after changing settings.
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionlingo.data import build_index                      # noqa: E402
from visionlingo.preprocessing import process_dataset           # noqa: E402
from visionlingo.preprocessing.mouth import default_detector    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    d, p = cfg["data"], cfg["preprocess"]
    df = build_index(ROOT / d["root"], d["subset"], d["frame_prefix"])
    print(f"Preprocessing {len(df)} clips -> {p['cache_dir']}/")

    detector = default_detector(p["face"]["scale_factor"], p["face"]["min_neighbors"], p["face"]["min_size"])
    out = process_dataset(
        df, ROOT / p["cache_dir"], p["img_size"], detector, p["probe_frames"],
        p["mouth"]["width_scale"], p["mouth"]["center_y"],
        {"side_frac": p["fallback"]["side_frac"], "center_x": p["fallback"]["center_x"],
         "center_y": p["fallback"]["center_y"]},
        d["frame_prefix"], overwrite=args.overwrite)

    errors = out[out["error"] != ""]
    fresh = out["detected"].dropna()
    print(f"\nDone. Cached clips: {len(out) - len(errors)} | errors: {len(errors)}")
    if len(fresh):
        rate = fresh.astype(bool).mean()
        print(f"Face detected in {rate:.1%} of newly processed clips; fallback box used for the rest.")
        if rate < 0.9:
            print("WARNING: many clips used the fallback box. Run scripts/visualize_preprocessing.py "
                  "and check the crops; the face/mouth settings in configs/config.yaml may need tuning.")
    if len(errors):
        print("\nFirst errors:\n", errors[["speaker", "word", "instance", "error"]].head(10).to_string(index=False))
    print(f"Index saved to {p['cache_dir']}/processed_index.csv")


if __name__ == "__main__":
    main()
