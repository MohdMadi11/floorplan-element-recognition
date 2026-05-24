from __future__ import annotations

from pathlib import Path

import cv2

from floorplan_ai.elements import DetectedElement


ELEMENT_STYLES = {
    "wall_candidate": {"color": (0, 80, 255), "label": "wall"},
    "column_candidate": {"color": (180, 30, 180), "label": "column"},
    "stair_candidate": {"color": (0, 165, 255), "label": "stairs"},
    "door_candidate": {"color": (40, 190, 40), "label": "door"},
    "window_candidate": {"color": (255, 80, 80), "label": "window"},
    "text_label": {"color": (80, 80, 255), "label": "text"},
    "elevator_candidate": {"color": (30, 30, 30), "label": "elev"},
}

DEFAULT_STYLE = {"color": (30, 160, 30), "label": "element"}


def annotate_elements(
    image_path: str | Path,
    elements: list[DetectedElement],
    out_image: str | Path,
) -> Path:
    image_path = Path(image_path)
    out_image = Path(out_image)
    out_image.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    overlay = image.copy()
    for element in elements:
        style = ELEMENT_STYLES.get(element.element_type, DEFAULT_STYLE)
        x, y, w, h = element.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), style["color"], thickness=2)

    annotated = cv2.addWeighted(overlay, 0.75, image, 0.25, 0)

    for element in elements:
        style = ELEMENT_STYLES.get(element.element_type, DEFAULT_STYLE)
        x, y, _, _ = element.bbox
        cv2.putText(
            annotated,
            style["label"],
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            style["color"],
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_image), annotated)
    return out_image
