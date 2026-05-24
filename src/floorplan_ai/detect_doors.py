from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from floorplan_ai.elements import DetectedElement
from floorplan_ai.geometry import dedupe_by_iou
from floorplan_ai.preprocess import preprocess_floorplan


def detect_door_candidates(
    image_path: str | Path,
    work_dir: str | Path,
) -> list[DetectedElement]:
    """Detect door candidates from swing arcs and compact line geometry."""
    image_path = Path(image_path)
    masks = preprocess_floorplan(image_path, work_dir)

    binary = cv2.imread(str(masks["binary"]), cv2.IMREAD_GRAYSCALE)
    line_mask = cv2.imread(str(masks["lines"]), cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise FileNotFoundError(f"Binary mask not found: {masks['binary']}")
    if line_mask is None:
        raise FileNotFoundError(f"Line mask not found: {masks['lines']}")

    arc_mask = cv2.bitwise_and(binary, cv2.bitwise_not(cv2.dilate(line_mask, np.ones((5, 5), np.uint8))))
    arc_mask = cv2.morphologyEx(
        arc_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    contours, _ = cv2.findContours(arc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[DetectedElement] = []

    for contour in contours:
        element = _contour_to_door(contour, line_mask)
        if element is not None:
            candidates.append(element)

    return _dedupe(candidates)


def _contour_to_door(contour: np.ndarray, line_mask: np.ndarray) -> DetectedElement | None:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 18 or h < 18:
        return None
    if w > 180 or h > 180:
        return None
    if max(w, h) < 42:
        return None
    if w * h < 1200:
        return None

    aspect_ratio = w / h
    if aspect_ratio < 0.35 or aspect_ratio > 2.8:
        return None

    area = cv2.contourArea(contour)
    bbox_area = w * h
    if bbox_area == 0:
        return None

    extent = area / bbox_area
    perimeter = cv2.arcLength(contour, closed=False)
    circularity = 0.0 if perimeter == 0 else 4 * np.pi * area / (perimeter * perimeter)

    # Door swings are curved and sparse; filled rectangles/text tend to have much higher extent.
    if extent > 0.34:
        return None
    if perimeter < 30:
        return None

    pad = 18
    y1 = max(0, y - pad)
    y2 = min(line_mask.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(line_mask.shape[1], x + w + pad)
    nearby_lines = int(np.count_nonzero(line_mask[y1:y2, x1:x2]))
    line_density = nearby_lines / max(1, (x2 - x1) * (y2 - y1))

    if line_density < 0.008:
        return None

    confidence = _score_door_candidate(extent=extent, circularity=circularity, line_density=line_density)
    return DetectedElement(
        element_type="door_candidate",
        bbox=(int(x), int(y), int(w), int(h)),
        confidence=confidence,
        source="swing_arc_component",
        metadata={
            "extent": round(float(extent), 3),
            "circularity": round(float(circularity), 3),
            "nearby_line_density": round(float(line_density), 4),
        },
    )


def _score_door_candidate(extent: float, circularity: float, line_density: float) -> float:
    arc_score = max(0.0, 1.0 - abs(extent - 0.18) / 0.25)
    line_score = min(1.0, line_density / 0.05)
    curve_score = min(1.0, circularity / 0.45)
    return max(0.3, min(0.9, 0.35 + 0.3 * arc_score + 0.2 * line_score + 0.15 * curve_score))


def _dedupe(elements: list[DetectedElement]) -> list[DetectedElement]:
    boxes = [element.bbox for element in elements]
    scores = [element.confidence for element in elements]
    kept_indexes = dedupe_by_iou(boxes, scores, threshold=0.2)
    kept = [elements[index] for index in kept_indexes]
    kept.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return kept
