from __future__ import annotations

from pathlib import Path
import shutil

import cv2
import numpy as np

from floorplan_ai.elements import DetectedElement
from floorplan_ai.geometry import dedupe_by_iou
from floorplan_ai.preprocess import preprocess_floorplan


def detect_text_labels(
    image_path: str | Path,
    work_dir: str | Path,
    enable_ocr: bool = True,
) -> list[DetectedElement]:
    """Detect text/room-label regions, with optional OCR when Tesseract exists."""
    image_path = Path(image_path)
    masks = preprocess_floorplan(image_path, work_dir)

    image = cv2.imread(str(image_path))
    binary = cv2.imread(str(masks["binary"]), cv2.IMREAD_GRAYSCALE)
    line_mask = cv2.imread(str(masks["lines"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    if binary is None or line_mask is None:
        raise FileNotFoundError("Preprocessing masks are missing.")

    text_mask = _build_text_mask(binary, line_mask)
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    elements: list[DetectedElement] = []
    for contour in contours:
        candidate = _contour_to_text_label(contour, image, enable_ocr=enable_ocr)
        if candidate is not None:
            elements.append(candidate)

    return _dedupe(elements)


def _build_text_mask(binary: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    line_dilation = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=1,
    )
    non_line = cv2.bitwise_and(binary, cv2.bitwise_not(line_dilation))
    non_line = cv2.morphologyEx(
        non_line,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    return cv2.morphologyEx(
        non_line,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6)),
    )


def _contour_to_text_label(
    contour: np.ndarray,
    image: np.ndarray,
    enable_ocr: bool,
) -> DetectedElement | None:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 16 or h < 8:
        return None
    if w > 900 or h > 260:
        return None

    aspect_ratio = w / h
    if aspect_ratio < 0.25 or aspect_ratio > 18:
        return None

    area = cv2.contourArea(contour)
    extent = area / max(1, w * h)
    if extent < 0.08 or extent > 0.85:
        return None

    text = _ocr_crop(image[y : y + h, x : x + w]) if enable_ocr else None
    confidence = _score_text_label(width=w, height=h, aspect_ratio=aspect_ratio, text=text)
    return DetectedElement(
        element_type="text_label",
        bbox=(int(x), int(y), int(w), int(h)),
        confidence=confidence,
        source="text_like_connected_component",
        metadata={
            "text": text,
            "aspect_ratio": round(float(aspect_ratio), 3),
            "extent": round(float(extent), 3),
        },
    )


def _ocr_crop(crop: np.ndarray) -> str | None:
    tesseract_cmd = _find_tesseract()
    if tesseract_cmd is None:
        return None
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(gray, config="--psm 7").strip()
        return text or None
    except Exception:
        return None


def _find_tesseract() -> str | None:
    path = shutil.which("tesseract")
    if path:
        return path

    common_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in common_paths:
        if candidate.exists():
            return str(candidate)
    return None


def _score_text_label(width: int, height: int, aspect_ratio: float, text: str | None) -> float:
    shape_score = 0.8 if aspect_ratio > 1.3 else 0.55
    size_score = min(1.0, max(width, height) / 120)
    ocr_bonus = 0.15 if text else 0.0
    return max(0.35, min(0.9, 0.35 + 0.25 * shape_score + 0.2 * size_score + ocr_bonus))


def _dedupe(elements: list[DetectedElement]) -> list[DetectedElement]:
    boxes = [element.bbox for element in elements]
    scores = [element.confidence for element in elements]
    kept_indexes = dedupe_by_iou(boxes, scores, threshold=0.2)
    kept = [elements[index] for index in kept_indexes]
    kept.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return kept
