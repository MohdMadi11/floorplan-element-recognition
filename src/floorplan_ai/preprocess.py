from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def preprocess_floorplan(image_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Create normalized masks from a rasterized floor plan image."""
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=35,
        C=12,
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    line_mask = cv2.bitwise_or(horizontal, vertical)

    stem = image_path.stem
    paths = {
        "gray": out_dir / f"{stem}_gray.png",
        "binary": out_dir / f"{stem}_binary.png",
        "lines": out_dir / f"{stem}_lines.png",
    }

    cv2.imwrite(str(paths["gray"]), gray)
    cv2.imwrite(str(paths["binary"]), binary)
    cv2.imwrite(str(paths["lines"]), line_mask)

    return paths


def image_stats(image_path: str | Path) -> dict[str, int | float]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    foreground = int(np.count_nonzero(image))
    total = int(image.size)
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "foreground_pixels": foreground,
        "foreground_ratio": foreground / total,
    }

