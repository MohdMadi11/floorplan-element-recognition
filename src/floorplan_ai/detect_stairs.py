from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from floorplan_ai.elements import DetectedElement
from floorplan_ai.preprocess import preprocess_floorplan


def detect_stair_candidates(
    image_path: str | Path,
    work_dir: str | Path,
    min_treads: int = 6,
) -> list[DetectedElement]:
    """Detect stair candidates as repeated groups of short parallel lines."""
    image_path = Path(image_path)
    masks = preprocess_floorplan(image_path, work_dir)

    line_mask = cv2.imread(str(masks["lines"]), cv2.IMREAD_GRAYSCALE)
    if line_mask is None:
        raise FileNotFoundError(f"Line mask not found: {masks['lines']}")

    horizontal = _find_stair_groups(
        line_mask,
        orientation="horizontal",
        line_kernel=(24, 1),
        group_kernel=(9, 95),
        min_treads=min_treads,
    )
    vertical = _find_stair_groups(
        line_mask,
        orientation="vertical",
        line_kernel=(1, 24),
        group_kernel=(95, 9),
        min_treads=min_treads,
    )

    return _dedupe(horizontal + vertical)


def _find_stair_groups(
    line_mask: np.ndarray,
    orientation: str,
    line_kernel: tuple[int, int],
    group_kernel: tuple[int, int],
    min_treads: int,
) -> list[DetectedElement]:
    tread_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, line_kernel)
    treads = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, tread_kernel)

    group = cv2.dilate(
        treads,
        cv2.getStructuringElement(cv2.MORPH_RECT, group_kernel),
        iterations=1,
    )
    group = cv2.morphologyEx(
        group,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
    )

    contours, _ = cv2.findContours(group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elements: list[DetectedElement] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 35 or h < 35:
            continue
        if w > 650 or h > 650:
            continue

        roi = treads[y : y + h, x : x + w]
        tread_count = _count_treads(roi, orientation)
        if tread_count < min_treads:
            continue

        density = float(np.count_nonzero(roi)) / float(max(1, w * h))
        if density < 0.025 or density > 0.35:
            continue

        confidence = _score_stair_candidate(tread_count=tread_count, density=density)
        elements.append(
            DetectedElement(
                element_type="stair_candidate",
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=confidence,
                source="parallel_tread_group",
                metadata={
                    "orientation": orientation,
                    "tread_count": int(tread_count),
                    "density": round(density, 4),
                },
            )
        )

    return elements


def _count_treads(roi: np.ndarray, orientation: str) -> int:
    projection = np.count_nonzero(roi, axis=1 if orientation == "horizontal" else 0)
    active = projection > max(4, int(projection.max() * 0.25)) if projection.size else []

    count = 0
    in_run = False
    for value in active:
        if value and not in_run:
            count += 1
            in_run = True
        elif not value:
            in_run = False
    return count


def _score_stair_candidate(tread_count: int, density: float) -> float:
    tread_score = min(1.0, tread_count / 12)
    density_score = max(0.0, 1.0 - abs(density - 0.08) / 0.12)
    return max(0.35, min(0.93, 0.45 + 0.35 * tread_score + 0.2 * density_score))


def _dedupe(elements: list[DetectedElement]) -> list[DetectedElement]:
    elements = sorted(elements, key=lambda item: item.confidence, reverse=True)
    kept: list[DetectedElement] = []
    for element in elements:
        if any(_iou(element.bbox, other.bbox) > 0.2 for other in kept):
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

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return 0.0 if union == 0 else intersection / union
