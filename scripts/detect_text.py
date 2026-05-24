from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorplan_ai.annotate import annotate_elements
from floorplan_ai.detect_text import detect_text_labels
from floorplan_ai.detect_walls import write_elements_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect text label regions from a floor plan image.")
    parser.add_argument("image", type=Path, help="Path to a rasterized floor plan image.")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs" / "preprocess")
    parser.add_argument("--out-image", type=Path, default=ROOT / "outputs" / "annotated" / "text.png")
    parser.add_argument("--out-json", type=Path, default=ROOT / "outputs" / "json" / "text.json")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR even if Tesseract is installed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    elements = detect_text_labels(args.image, work_dir=args.work_dir, enable_ocr=not args.no_ocr)
    annotated_path = annotate_elements(args.image, elements, args.out_image)
    json_path = write_elements_json(elements, args.out_json, args.image)
    print(f"detected text labels: {len(elements)}")
    print(f"annotated image: {annotated_path}")
    print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
