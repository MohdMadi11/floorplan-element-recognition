from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorplan_ai.detect_walls import annotate_walls, detect_wall_candidates, write_elements_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect wall candidates from a floor plan image.")
    parser.add_argument("image", type=Path, help="Path to a rasterized floor plan image.")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs" / "preprocess")
    parser.add_argument("--out-image", type=Path, default=ROOT / "outputs" / "annotated" / "walls.png")
    parser.add_argument("--out-json", type=Path, default=ROOT / "outputs" / "json" / "walls.json")
    parser.add_argument("--min-length", type=int, default=90, help="Minimum candidate wall length in pixels.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    elements = detect_wall_candidates(
        args.image,
        work_dir=args.work_dir,
        min_length=args.min_length,
    )
    annotated_path = annotate_walls(args.image, elements, args.out_image)
    json_path = write_elements_json(elements, args.out_json, args.image)

    print(f"detected wall candidates: {len(elements)}")
    print(f"annotated image: {annotated_path}")
    print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
