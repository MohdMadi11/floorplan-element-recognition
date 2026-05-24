from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from floorplan_ai.elements import DetectedElement
from floorplan_ai.geometry import dedupe_by_iou
from floorplan_ai.preprocess import preprocess_floorplan


def detect_elevator_candidates(
    image_path: str | Path,
    work_dir: str | Path,
) -> list[DetectedElement]:
    """Detect elevator/lift-shaft candidates as boxed regions with diagonal/X detail."""
    image_path = Path(image_path)
    masks = preprocess_floorplan(image_path, work_dir)

    line_mask = cv2.imread(str(masks["lines"]), cv2.IMREAD_GRAYSCALE)
    binary = cv2.imread(str(masks["binary"]), cv2.IMREAD_GRAYSCALE)
    if line_mask is None or binary is None:
        raise FileNotFoundError("Preprocessing masks are missing.")

    structural = cv2.bitwise_or(line_mask, binary)
    closed = cv2.morphologyEx(
        structural,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    elements: list[DetectedElement] = []
    for contour in contours:
        candidate = _contour_to_elevator(contour, binary)
        if candidate is not None:
            elements.append(candidate)

    return _dedupe(elements)


def _contour_to_elevator(contour: np.ndarray, binary: np.ndarray) -> DetectedElement | None:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 28 or h < 28:
        return None
    if w > 320 or h > 320:
        return None

    aspect_ratio = w / h
    if aspect_ratio < 0.35 or aspect_ratio > 2.8:
        return None

    roi = binary[y : y + h, x : x + w]
    diagonal_count = _count_diagonal_lines(roi)
    density = float(np.count_nonzero(roi)) / float(max(1, w * h))
    if diagonal_count < 1:
        return None
    if density < 0.025 or density > 0.55:
        return None

    confidence = _score_elevator(diagonal_count=diagonal_count, density=density, aspect_ratio=aspect_ratio)
    return DetectedElement(
        element_type="elevator_candidate",
        bbox=(int(x), int(y), int(w), int(h)),
        confidence=confidence,
        source="boxed_diagonal_component",
        metadata={
            "diagonal_count": int(diagonal_count),
            "density": round(density, 4),
            "aspect_ratio": round(float(aspect_ratio), 3),
        },
    )


def _count_diagonal_lines(roi: np.ndarray) -> int:
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=18,
        minLineLength=max(12, min(roi.shape[:2]) // 3),
        maxLineGap=6,
    )
    if lines is None:
        return 0

    count = 0
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = angle if angle <= 90 else 180 - angle
        if 25 <= angle <= 65:
            count += 1
    return count


def _score_elevator(diagonal_count: int, density: float, aspect_ratio: float) -> float:
    diagonal_score = min(1.0, diagonal_count / 2)
    density_score = max(0.0, 1.0 - abs(density - 0.12) / 0.18)
    shape_score = max(0.0, 1.0 - abs(1.0 - aspect_ratio) / 1.2)
    return max(0.35, min(0.9, 0.35 + 0.25 * diagonal_score + 0.2 * density_score + 0.2 * shape_score))


def _dedupe(elements: list[DetectedElement]) -> list[DetectedElement]:
    boxes = [element.bbox for element in elements]
    scores = [element.confidence for element in elements]
    kept_indexes = dedupe_by_iou(boxes, scores, threshold=0.2)
    kept = [elements[index] for index in kept_indexes]
    kept.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return kept
