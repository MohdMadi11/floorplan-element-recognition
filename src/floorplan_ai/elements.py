from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
from typing import Any


@dataclass(frozen=True)
class DetectedElement:
    """Shared JSON-friendly representation for detected floor plan elements."""

    element_type: str
    bbox: tuple[int, int, int, int]
    confidence: float
    source: str
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = data.pop("element_type")
        data["bbox"] = list(self.bbox)
        data["confidence"] = round(self.confidence, 3)
        return data


def count_by_type(elements: list[DetectedElement]) -> dict[str, int]:
    return dict(Counter(element.element_type for element in elements))
