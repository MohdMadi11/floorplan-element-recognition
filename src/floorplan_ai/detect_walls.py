from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from floorplan_ai.annotate import annotate_elements
from floorplan_ai.elements import DetectedElement, count_by_type
from floorplan_ai.preprocess import preprocess_floorplan


WALL_COLOR = (0, 80, 255)


def detect_wall_candidates(
    image_path: str | Path,
    work_dir: str | Path,
    min_length: int = 90,
    max_thickness: int = 28,
) -> list[DetectedElement]:
    """Detect long axis-aligned wall candidates from a floor plan image.

    This is an early geometric detector, not the final semantic wall model.
    It intentionally favors recall: later steps will suppress dimension lines,
    stairs, symbols, and furniture-like false positives.
    """
    image_path = Path(image_path)
    work_dir = Path(work_dir)
    masks = preprocess_floorplan(image_path, work_dir)

    line_mask = cv2.imread(str(masks["lines"]), cv2.IMREAD_GRAYSCALE)
    if line_mask is None:
        raise FileNotFoundError(f"Line mask not found: {masks['lines']}")

    horizontal = _extract_axis_components(
        line_mask,
        kernel_size=(75, 1),
        min_length=min_length,
        max_thickness=max_thickness,
        orientation="horizontal",
    )
    vertical = _extract_axis_components(
        line_mask,
        kernel_size=(1, 75),
        min_length=min_length,
        max_thickness=max_thickness,
        orientation="vertical",
    )

    return horizontal + vertical


def annotate_walls(
    image_path: str | Path,
    elements: list[DetectedElement],
    out_image: str | Path,
) -> Path:
    return annotate_elements(image_path, elements, out_image)


def write_elements_json(
    elements: list[DetectedElement],
    out_json: str | Path,
    image_path: str | Path,
    *,
    source_pdf: str | Path | None = None,
    page_number: int | None = None,
) -> Path:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pipeline_version": "0.1.0",
        "source_pdf": str(source_pdf) if source_pdf is not None else None,
        "image": str(image_path),
        "page_number": page_number,
        "element_count": len(elements),
        "counts_by_type": count_by_type(elements),
        "elements": [element.to_json() for element in elements],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_json


def _extract_axis_components(
    mask: np.ndarray,
    kernel_size: tuple[int, int],
    min_length: int,
    max_thickness: int,
    orientation: str,
) -> list[DetectedElement]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    axis_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    axis_mask = cv2.dilate(axis_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(axis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elements: list[DetectedElement] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = w if orientation == "horizontal" else h
        thickness = h if orientation == "horizontal" else w

        if length < min_length:
            continue
        if thickness > max_thickness:
            continue

        confidence = _score_wall_candidate(length=length, thickness=thickness)
        elements.append(
            DetectedElement(
                element_type="wall_candidate",
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=confidence,
                source="axis_aligned_line_component",
                metadata={
                    "orientation": orientation,
                    "length_px": int(length),
                    "thickness_px": int(thickness),
                },
            )
        )

    elements.sort(key=lambda item: (item.metadata["orientation"], item.bbox[1], item.bbox[0]))
    return elements


def _score_wall_candidate(length: int, thickness: int) -> float:
    length_score = min(1.0, length / 800)
    thickness_score = 1.0 - min(0.5, max(0, thickness - 8) / 40)
    return max(0.35, min(0.95, 0.45 + 0.35 * length_score + 0.2 * thickness_score))
