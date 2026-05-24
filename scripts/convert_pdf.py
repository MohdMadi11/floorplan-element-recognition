from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorplan_ai.pdf_convert import PdfConversionError, convert_pdf_to_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a floor plan PDF to PNG page images.")
    parser.add_argument("pdf", type=Path, help="Path to the input PDF.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "images", help="Output directory.")
    parser.add_argument("--dpi", type=int, default=250, help="Rasterization DPI.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        images = convert_pdf_to_images(args.pdf, args.out, dpi=args.dpi)
    except (FileNotFoundError, PdfConversionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for image in images:
        print(image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
