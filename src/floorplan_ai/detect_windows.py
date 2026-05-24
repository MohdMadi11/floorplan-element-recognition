from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from floorplan_ai.elements import DetectedElement
from floorplan_ai.geometry import dedupe_by_iou
from floorplan_ai.preprocess import preprocess_floorplan


def detect_window_candidates(
    image_path: str | Path,
    work_dir: str | Path,
) -> list[DetectedElement]:
    """Detect window candidates as short paired line segments near walls."""
    image_path = Path(image_path)
    masks = preprocess_floorplan(image_path, work_dir)

    line_mask = cv2.imread(str(masks["lines"]), cv2.IMREAD_GRAYSCALE)
    if line_mask is None:
        raise FileNotFoundError(f"Line mask not found: {masks['lines']}")

    horizontal = _find_window_groups(
        line_mask,
        orientation="horizontal",
        short_kernel=(28, 1),
        pair_kernel=(42, 9),
    )
    vertical = _find_window_groups(
        line_mask,
        orientation="vertical",
        short_kernel=(1, 28),
        pair_kernel=(9, 42),
    )
    return _dedupe(horizontal + vertical)


def _find_window_groups(
    line_mask: np.ndarray,
    orientation: str,
    short_kernel: tuple[int, int],
    pair_kernel: tuple[int, int],
) -> list[DetectedElement]:
    short_lines = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, short_kernel),
    )
    paired = cv2.dilate(
        short_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, pair_kernel),
        iterations=1,
    )

    contours, _ = cv2.findContours(paired, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elements: list[DetectedElement] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        long_side = w if orientation == "horizontal" else h
        short_side = h if orientation == "horizontal" else w

        if long_side < 45 or long_side > 230:
            continue
        if short_side < 8 or short_side > 55:
            continue

        roi = short_lines[y : y + h, x : x + w]
        stroke_count = _count_strokes(roi, orientation)
        if stroke_count < 2 or stroke_count > 4:
            continue

        density = float(np.count_nonzero(roi)) / float(max(1, w * h))
        if density < 0.025 or density > 0.5:
            continue

        confidence = _score_window_candidate(stroke_count=stroke_count, density=density, long_side=long_side)
        elements.append(
            DetectedElement(
                element_type="window_candidate",
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=confidence,
                source="paired_short_line_group",
                metadata={
                    "orientation": orientation,
                    "stroke_count": int(stroke_count),
                    "density": round(density, 4),
                    "length_px": int(long_side),
                },
            )
        )

    return elements


def _count_strokes(roi: np.ndarray, orientation: str) -> int:
    projection = np.count_nonzero(roi, axis=1 if orientation == "horizontal" else 0)
    if projection.size == 0 or projection.max() == 0:
        return 0

    active = projection > max(3, int(projection.max() * 0.3))
    count = 0
    in_run = False
    for value in active:
        if value and not in_run:
            count += 1
            in_run = True
        elif not value:
            in_run = False
    return count


def _score_window_candidate(stroke_count: int, density: float, long_side: int) -> float:
    pair_score = 1.0 if stroke_count in (2, 3) else 0.75
    density_score = max(0.0, 1.0 - abs(density - 0.14) / 0.18)
    length_score = min(1.0, long_side / 120)
    return max(0.3, min(0.88, 0.35 + 0.25 * pair_score + 0.2 * density_score + 0.15 * length_score))


def _dedupe(elements: list[DetectedElement]) -> list[DetectedElement]:
    boxes = [element.bbox for element in elements]
    scores = [element.confidence for element in elements]
    kept_indexes = dedupe_by_iou(boxes, scores, threshold=0.25)
    kept = [elements[index] for index in kept_indexes]
    kept.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return kept
