from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorplan_ai.pipeline import run_pdf_pipeline
from floorplan_ai.elements import count_by_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the floor plan recognition pipeline on a PDF."
    )
    parser.add_argument("pdf", type=Path, help="Path to the input floor plan PDF.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs", help="Output root directory.")
    parser.add_argument("--dpi", type=int, default=200, help="PDF rasterization DPI.")
    parser.add_argument(
        "--min-wall-length",
        type=int,
        default=90,
        help="Minimum wall-candidate length in pixels.",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Detect text boxes but skip OCR text extraction.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_pdf_pipeline(
        args.pdf,
        output_root=args.out,
        dpi=args.dpi,
        min_wall_length=args.min_wall_length,
        enable_ocr=not args.no_ocr,
    )

    print(f"processed PDF: {args.pdf}")
    for result in results:
        print(f"page {result.page_number}:")
        print(f"  image: {result.image_path}")
        print(f"  annotated: {result.annotated_path}")
        print(f"  json: {result.json_path}")
        print(f"  elements: {len(result.elements)}")
        print(f"  counts: {count_by_type(result.elements)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
