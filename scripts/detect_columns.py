from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorplan_ai.annotate import annotate_elements
from floorplan_ai.detect_columns import detect_column_candidates
from floorplan_ai.detect_walls import write_elements_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect column candidates from a floor plan image.")
    parser.add_argument("image", type=Path, help="Path to a rasterized floor plan image.")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs" / "preprocess")
    parser.add_argument("--out-image", type=Path, default=ROOT / "outputs" / "annotated" / "columns.png")
    parser.add_argument("--out-json", type=Path, default=ROOT / "outputs" / "json" / "columns.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    elements = detect_column_candidates(args.image, work_dir=args.work_dir)
    annotated_path = annotate_elements(args.image, elements, args.out_image)
    json_path = write_elements_json(elements, args.out_json, args.image)

    print(f"detected column candidates: {len(elements)}")
    print(f"annotated image: {annotated_path}")
    print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
