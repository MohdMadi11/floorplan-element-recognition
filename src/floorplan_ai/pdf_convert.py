from __future__ import annotations

from pathlib import Path


class PdfConversionError(RuntimeError):
    """Raised when no available backend can convert a PDF to images."""


def convert_pdf_to_images(pdf_path: str | Path, out_dir: str | Path, dpi: int = 250) -> list[Path]:
    """Convert each page of a PDF into a PNG image.

    The preferred backend is pypdfium2 because it is self-contained and does not
    require Poppler. We keep the function small and backend-focused so later
    pipeline stages can treat PDF conversion as a solved input step.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        return _convert_with_pdfium(pdf_path, out_dir, dpi)
    except ModuleNotFoundError as exc:
        raise PdfConversionError(
            "PDF conversion needs pypdfium2. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc


def _convert_with_pdfium(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    import pypdfium2 as pdfium

    scale = dpi / 72
    pdf = pdfium.PdfDocument(str(pdf_path))
    output_paths: list[Path] = []

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        output_path = out_dir / f"{pdf_path.stem}_page_{page_index + 1}.png"
        image.save(output_path)
        output_paths.append(output_path)

    return output_paths

