from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from floorplan_ai.annotate import annotate_elements
from floorplan_ai.detect_columns import detect_column_candidates
from floorplan_ai.detect_doors import detect_door_candidates
from floorplan_ai.detect_elevators import detect_elevator_candidates
from floorplan_ai.detect_stairs import detect_stair_candidates
from floorplan_ai.detect_text import detect_text_labels
from floorplan_ai.detect_walls import detect_wall_candidates, write_elements_json
from floorplan_ai.detect_windows import detect_window_candidates
from floorplan_ai.elements import DetectedElement
from floorplan_ai.pdf_convert import convert_pdf_to_images
from floorplan_ai.postprocess import postprocess_elements


@dataclass(frozen=True)
class PipelinePageResult:
    page_number: int
    image_path: Path
    annotated_path: Path
    json_path: Path
    elements: list[DetectedElement]


def detect_all_candidates(
    image_path: Path,
    preprocess_dir: Path,
    min_wall_length: int,
    enable_ocr: bool,
) -> list[DetectedElement]:
    """Run every currently implemented architectural element detector."""
    elements = detect_wall_candidates(
        image_path,
        work_dir=preprocess_dir,
        min_length=min_wall_length,
    )
    elements.extend(detect_column_candidates(image_path, work_dir=preprocess_dir))
    elements.extend(detect_stair_candidates(image_path, work_dir=preprocess_dir))
    elements.extend(detect_door_candidates(image_path, work_dir=preprocess_dir))
    elements.extend(detect_window_candidates(image_path, work_dir=preprocess_dir))
    elements.extend(detect_text_labels(image_path, work_dir=preprocess_dir, enable_ocr=enable_ocr))
    elements.extend(detect_elevator_candidates(image_path, work_dir=preprocess_dir))
    return postprocess_elements(elements)


def run_pdf_pipeline(
    pdf_path: str | Path,
    output_root: str | Path = "outputs",
    dpi: int = 200,
    min_wall_length: int = 90,
    enable_ocr: bool = True,
) -> list[PipelinePageResult]:
    """Run the current floor plan recognition pipeline for every page in a PDF."""
    pdf_path = Path(pdf_path)
    output_root = Path(output_root)

    image_dir = output_root / "images"
    preprocess_dir = output_root / "preprocess"
    annotated_dir = output_root / "annotated"
    json_dir = output_root / "json"

    image_paths = convert_pdf_to_images(pdf_path, image_dir, dpi=dpi)
    results: list[PipelinePageResult] = []

    for page_number, image_path in enumerate(image_paths, start=1):
        output_stem = f"{pdf_path.stem}_page_{page_number}"
        elements = detect_all_candidates(image_path, preprocess_dir, min_wall_length, enable_ocr)

        annotated_path = annotated_dir / f"{output_stem}_annotated.png"
        json_path = json_dir / f"{output_stem}_elements.json"

        annotate_elements(image_path, elements, annotated_path)
        write_elements_json(
            elements,
            json_path,
            image_path,
            source_pdf=pdf_path,
            page_number=page_number,
        )

        results.append(
            PipelinePageResult(
                page_number=page_number,
                image_path=image_path,
                annotated_path=annotated_path,
                json_path=json_path,
                elements=elements,
            )
        )

    return results
