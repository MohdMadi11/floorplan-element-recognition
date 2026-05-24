from __future__ import annotations

from floorplan_ai.elements import DetectedElement
from floorplan_ai.geometry import expand_box, intersection_over_first, iou


TEXT_SUPPRESSED_TYPES = {"column_candidate", "window_candidate", "stair_candidate"}
CONFLICT_PRIORITY = {
    "elevator_candidate": 7,
    "door_candidate": 6,
    "text_label": 5,
    "stair_candidate": 3,
    "window_candidate": 2,
    "column_candidate": 1,
    "wall_candidate": 0,
}


def postprocess_elements(elements: list[DetectedElement]) -> list[DetectedElement]:
    elements = elements + _derive_stairs_from_wall_treads(elements)
    elements = _remove_text_overlaps(elements)
    elements = _remove_walls_in_text_dense_regions(elements)
    elements = _remove_walls_inside_stairs(elements)
    elements = _resolve_same_area_conflicts(elements)
    return sorted(elements, key=lambda item: (item.element_type, item.bbox[1], item.bbox[0]))


def _remove_text_overlaps(elements: list[DetectedElement]) -> list[DetectedElement]:
    text_boxes = [_expanded_text_box(element.bbox) for element in elements if element.element_type == "text_label"]
    kept: list[DetectedElement] = []

    for element in elements:
        if element.element_type in TEXT_SUPPRESSED_TYPES:
            if any(intersection_over_first(element.bbox, text_box) > 0.22 for text_box in text_boxes):
                continue
        if element.element_type == "wall_candidate" and _wall_is_text_like(element):
            if any(intersection_over_first(element.bbox, text_box) > 0.55 for text_box in text_boxes):
                continue
        kept.append(element)

    return kept


def _remove_walls_inside_stairs(elements: list[DetectedElement]) -> list[DetectedElement]:
    stair_boxes = [
        expand_box(element.bbox, 14, 14)
        for element in elements
        if element.element_type == "stair_candidate"
    ]
    kept: list[DetectedElement] = []

    for element in elements:
        if element.element_type == "wall_candidate" and _wall_is_stair_tread_like(element):
            if any(intersection_over_first(element.bbox, stair_box) > 0.35 for stair_box in stair_boxes):
                continue
        kept.append(element)

    return kept


def _remove_walls_in_text_dense_regions(elements: list[DetectedElement]) -> list[DetectedElement]:
    text_boxes = [element.bbox for element in elements if element.element_type == "text_label"]
    kept: list[DetectedElement] = []

    for element in elements:
        if element.element_type == "wall_candidate" and _wall_is_text_like_or_table_like(element):
            nearby_text_count = sum(
                1 for text_box in text_boxes if _boxes_near(element.bbox, text_box, pad_x=170, pad_y=120)
            )
            if nearby_text_count >= 3:
                continue
        kept.append(element)

    return kept


def _resolve_same_area_conflicts(elements: list[DetectedElement]) -> list[DetectedElement]:
    ordered = sorted(
        elements,
        key=lambda item: (CONFLICT_PRIORITY.get(item.element_type, 0), item.confidence),
        reverse=True,
    )
    kept: list[DetectedElement] = []

    for element in ordered:
        if element.element_type == "wall_candidate":
            kept.append(element)
            continue

        if any(
            iou(element.bbox, other.bbox) > 0.35
            and CONFLICT_PRIORITY.get(other.element_type, 0) >= CONFLICT_PRIORITY.get(element.element_type, 0)
            for other in kept
            if other.element_type != "wall_candidate"
        ):
            continue
        kept.append(element)

    return kept


def _is_small(bbox: tuple[int, int, int, int]) -> bool:
    _, _, w, h = bbox
    return w < 80 and h < 80


def _expanded_text_box(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    _, _, w, h = bbox
    return expand_box(
        bbox,
        pad_x=max(16, int(w * 0.45)),
        pad_y=max(24, int(h * 5)),
    )


def _wall_is_text_like(element: DetectedElement) -> bool:
    _, _, w, h = element.bbox
    length = max(w, h)
    thickness = min(w, h)
    return length <= 360 and thickness <= 12


def _wall_is_stair_tread_like(element: DetectedElement) -> bool:
    _, _, w, h = element.bbox
    length = max(w, h)
    thickness = min(w, h)
    return 25 <= length <= 650 and thickness <= 18


def _wall_is_text_like_or_table_like(element: DetectedElement) -> bool:
    _, _, w, h = element.bbox
    length = max(w, h)
    thickness = min(w, h)
    return length <= 1100 and thickness <= 14


def _boxes_near(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad_x: int,
    pad_y: int,
) -> bool:
    return iou(expand_box(a, pad_x, pad_y), b) > 0


def _derive_stairs_from_wall_treads(elements: list[DetectedElement]) -> list[DetectedElement]:
    walls = [element for element in elements if element.element_type == "wall_candidate"]
    derived = []
    derived.extend(_derive_axis_stairs(walls, orientation="vertical"))
    derived.extend(_derive_axis_stairs(walls, orientation="horizontal"))
    return derived


def _derive_axis_stairs(walls: list[DetectedElement], orientation: str) -> list[DetectedElement]:
    if orientation == "vertical":
        treads = [wall for wall in walls if 35 <= wall.bbox[3] <= 520 and wall.bbox[2] <= 18]
        treads.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
        return _cluster_parallel_treads(treads, orientation)

    treads = [wall for wall in walls if 35 <= wall.bbox[2] <= 520 and wall.bbox[3] <= 18]
    treads.sort(key=lambda item: (item.bbox[0], item.bbox[1]))
    return _cluster_parallel_treads(treads, orientation)


def _cluster_parallel_treads(treads: list[DetectedElement], orientation: str) -> list[DetectedElement]:
    clusters: list[list[DetectedElement]] = []
    used: set[int] = set()

    for index, tread in enumerate(treads):
        if index in used:
            continue

        cluster = [tread]
        used.add(index)
        for other_index, other in enumerate(treads[index + 1 :], start=index + 1):
            if other_index in used:
                continue
            if _treads_compatible(tread.bbox, other.bbox, orientation):
                cluster.append(other)
                used.add(other_index)

        if len(cluster) >= 4:
            clusters.append(cluster)

    return [_cluster_to_stair(cluster, orientation) for cluster in clusters]


def _treads_compatible(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    orientation: str,
) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    if orientation == "vertical":
        y_close = abs(ay - by) <= 70 and abs((ay + ah) - (by + bh)) <= 90
        x_spacing = 8 <= abs(ax - bx) <= 95
        similar_length = abs(ah - bh) <= max(40, int(max(ah, bh) * 0.45))
        return y_close and x_spacing and similar_length

    x_close = abs(ax - bx) <= 70 and abs((ax + aw) - (bx + bw)) <= 90
    y_spacing = 8 <= abs(ay - by) <= 95
    similar_length = abs(aw - bw) <= max(40, int(max(aw, bw) * 0.45))
    return x_close and y_spacing and similar_length


def _cluster_to_stair(cluster: list[DetectedElement], orientation: str) -> DetectedElement:
    x1 = min(item.bbox[0] for item in cluster)
    y1 = min(item.bbox[1] for item in cluster)
    x2 = max(item.bbox[0] + item.bbox[2] for item in cluster)
    y2 = max(item.bbox[1] + item.bbox[3] for item in cluster)
    tread_count = len(cluster)
    confidence = max(0.45, min(0.88, 0.45 + 0.07 * min(tread_count, 6)))
    return DetectedElement(
        element_type="stair_candidate",
        bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
        confidence=confidence,
        source="parallel_wall_tread_cluster",
        metadata={
            "orientation": orientation,
            "tread_count": int(tread_count),
        },
    )
