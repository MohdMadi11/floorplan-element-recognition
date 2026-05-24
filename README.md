# Architectural Element Recognition from PDF Floor Plans

This project implements a classical computer vision pipeline for detecting architectural elements in PDF floor plans.

It accepts a PDF, converts it to images internally, produces annotated PNG outputs, and writes structured JSON detections.

## Detected Elements

The current pipeline outputs these candidate types:

- `wall_candidate`
- `column_candidate`
- `stair_candidate`
- `door_candidate`
- `window_candidate`
- `text_label`
- `elevator_candidate`

The word `candidate` is intentional: this is a high-recall CV pipeline designed to work across different drawing styles without labeled training data.

## Setup

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

OCR is optional but recommended for text labels. On Windows:

```powershell
winget install --id UB-Mannheim.TesseractOCR --source winget
```

The code also checks the common install path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Run The Pipeline

Use `250 DPI` for the assignment tests:

```powershell
python scripts/run_pipeline.py "data/floorplan1.pdf" --out outputs --dpi 250
```

Faster structural run without OCR:

```powershell
python scripts/run_pipeline.py "data/floorplan1.pdf" --out outputs --dpi 250 --no-ocr
```

Expected outputs:

- `outputs/images/floorplan1_page_1.png`
- `outputs/preprocess/floorplan1_page_1_binary.png`
- `outputs/preprocess/floorplan1_page_1_lines.png`
- `outputs/annotated/floorplan1_page_1_annotated.png`
- `outputs/json/floorplan1_page_1_elements.json`

## Final Test Commands

All tests should use the same output root and `250 DPI`:

```powershell
python scripts/run_pipeline.py "data/floorplan1.pdf" --out outputs --dpi 250
python scripts/run_pipeline.py "data/floorplan2.pdf" --out outputs --dpi 250
python scripts/run_pipeline.py "data/floorplan3.pdf" --out outputs --dpi 250
python scripts/run_pipeline.py "data/floorplan4.pdf" --out outputs --dpi 250
```

For a new company-provided PDF:

```powershell
python scripts/run_pipeline.py "data/company_test.pdf" --out outputs --dpi 250
```

Then inspect:

- `outputs/annotated/company_test_page_1_annotated.png`
- `outputs/json/company_test_page_1_elements.json`

If OCR is too slow on a very large PDF:

```powershell
python scripts/run_pipeline.py "data/company_test.pdf" --out outputs --dpi 250 --no-ocr
```

## VS Code Tasks

Press `Ctrl+Shift+P`, choose `Tasks: Run Task`, then select:

- `Run Pipeline: floorplan1`
- `Run Pipeline: floorplan2 sanity check`
- `Run Pipeline: floorplan3 validation`
- `Run Pipeline: floorplan4 final test`
- `Run Pipeline: floorplan1 no OCR`

The lower-level debug tasks are still available for individual detectors.

## Pipeline Summary

1. `pypdfium2` rasterizes the PDF into a PNG.
2. OpenCV converts the image to grayscale.
3. Adaptive thresholding creates a binary drawing mask.
4. Morphological filters extract horizontal/vertical line masks.
5. Detector modules identify element candidates:
   - walls: long axis-aligned line components
   - columns: compact circle/square components
   - stairs: repeated parallel tread groups
   - doors: sparse arc-like components near line geometry
   - windows: short paired line groups
   - text labels: grouped text-like connected components, optionally OCR'd
   - elevators: boxed regions with diagonal/X-like internal geometry
6. Post-processing reduces overlaps and obvious text-related false positives.
7. The system writes one annotated image and one JSON file per PDF page.

## Validation Workflow

For submission, run the pipeline on the provided PDFs at `250 DPI` and inspect both outputs:

- the annotated PNG for visual correctness
- the JSON file for structured detections

Keep one PDF as a final holdout test while tuning. For a company-provided unseen PDF, place it in `data/company_test.pdf` and run:

```powershell
python scripts/run_pipeline.py "data/company_test.pdf" --out outputs --dpi 250
```

## Known Limitations

- This is a heuristic CV pipeline, not a trained detector.
- Doors are intentionally high-recall and can include arc-like fixtures.
- OCR is useful for label regions but can be noisy on tiny or rotated text.
- Elevator/lift shaft detection depends heavily on whether the drawing uses boxed diagonal/X symbols.
- Candidate outputs are meant to be inspectable and extensible rather than final production-grade detections.
