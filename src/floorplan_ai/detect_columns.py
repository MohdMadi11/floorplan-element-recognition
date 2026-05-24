from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from floorplan_ai.elements import DetectedElement
from floorplan_ai.preprocess import preprocess_floorplan


def detect_column_candidates(
    image_path: str | Path,
    work_dir: str | Path,
    min_size: int = 8,
    max_size: int = 90,
) -> list[DetectedElement]:
    """Detect compact square/circle-like column candidates.

    This is a high-recall detector. It looks for architectural column symbols
    that are compact and roughly square, while avoiding long wall and stair
    geometry. Text suppression will improve this in a later stage.
    """
    image_path = Path(image_path)
    masks = preprocess_floorplan(image_path, work_dir)

    binary = cv2.imread(str(masks["binary"]), cv2.IMREAD_GRAYSCALE)
    line_mask = cv2.imread(str(masks["lines"]), cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise FileNotFoundError(f"Binary mask not found: {masks['binary']}")
    if line_mask is None:
        raise FileNotFoundError(f"Line mask not found: {masks['lines']}")

    compact_mask = _build_compact_component_mask(binary, line_mask)
    contours, _ = cv2.findContours(compact_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    elements: list[DetectedElement] = []
    for contour in contours:
        candidate = _contour_to_column_candidate(contour, min_size=min_size, max_size=max_size)
        if candidate is not None:
            elements.append(candidate)

    return _dedupe_nearby_candidates(elements)


def _build_compact_component_mask(binary: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    line_dilation = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=1,
    )
    non_line = cv2.bitwise_and(binary, cv2.bitwise_not(line_dilation))

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(non_line, cv2.MORPH_CLOSE, close_kernel)


def _contour_to_column_candidate(
    contour: np.ndarray,
    min_size: int,
    max_size: int,
) -> DetectedElement | None:
    x, y, w, h = cv2.boundingRect(contour)
    if w < min_size or h < min_size:
        return None
    if w > max_size or h > max_size:
        return None

    aspect_ratio = w / h
    if aspect_ratio < 0.55 or aspect_ratio > 1.8:
        return None

    area = cv2.contourArea(contour)
    bbox_area = w * h
    if bbox_area == 0:
        return None

    extent = area / bbox_area
    perimeter = cv2.arcLength(contour, closed=True)
    circularity = 0.0 if perimeter == 0 else 4 * np.pi * area / (perimeter * perimeter)

    if extent < 0.18:
        return None
    if circularity < 0.18 and extent < 0.45:
        return None

    confidence = _score_column_candidate(
        aspect_ratio=aspect_ratio,
        extent=extent,
        circularity=circularity,
        size=max(w, h),
    )
    return DetectedElement(
        element_type="column_candidate",
        bbox=(int(x), int(y), int(w), int(h)),
        confidence=confidence,
        source="compact_shape_component",
        metadata={
            "aspect_ratio": round(float(aspect_ratio), 3),
            "extent": round(float(extent), 3),
            "circularity": round(float(circularity), 3),
            "size_px": int(max(w, h)),
        },
    )


def _score_column_candidate(
    aspect_ratio: float,
    extent: float,
    circularity: float,
    size: int,
) -> float:
    square_score = max(0.0, 1.0 - abs(1.0 - aspect_ratio))
    compactness = min(1.0, (extent + circularity) / 1.4)
    size_score = min(1.0, size / 45)
    return max(0.3, min(0.92, 0.35 + 0.25 * square_score + 0.25 * compactness + 0.15 * size_score))


def _dedupe_nearby_candidates(elements: list[DetectedElement]) -> list[DetectedElement]:
    elements = sorted(elements, key=lambda item: item.confidence, reverse=True)
    kept: list[DetectedElement] = []

    for element in elements:
        if any(_iou(element.bbox, other.bbox) > 0.25 for other in kept):
            continue
        kept.append(element)

    kept.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return 0.0 if union == 0 else intersection / union

