from __future__ import annotations


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return 0.0 if union == 0 else intersection / union


def intersection_over_first(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(1, aw * ah)
    return intersection / first_area


def expand_box(
    bbox: tuple[int, int, int, int],
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return (x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y)


def dedupe_by_iou(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    threshold: float = 0.25,
) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda index: scores[index], reverse=True)
    kept: list[int] = []

    for index in order:
        if any(iou(boxes[index], boxes[kept_index]) > threshold for kept_index in kept):
            continue
        kept.append(index)

    return kept
