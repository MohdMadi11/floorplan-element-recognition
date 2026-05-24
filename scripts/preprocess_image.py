from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorplan_ai.preprocess import image_stats, preprocess_floorplan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create binary and line masks from a floor plan image.")
    parser.add_argument("image", type=Path, help="Path to a rasterized floor plan image.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "preprocess", help="Output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = preprocess_floorplan(args.image, args.out)

    for name, path in paths.items():
        stats = image_stats(path)
        print(f"{name}: {path} ({stats['width']}x{stats['height']}, foreground={stats['foreground_ratio']:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
