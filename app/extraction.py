from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
from typing import Any

import fitz
import httpx

from app.config import Settings


@dataclass
class ExtractedLine:
    bbox: tuple[float, float, float, float] | None
    font_name: str | None
    font_size: float | None
    is_bold: bool
    page_number: int
    text: str


@dataclass
class ExtractedPage:
    extractor: str
    lines: list[ExtractedLine]
    page_number: int
    text: str


def validate_pdf_file(path: Path, max_pdf_mb: int) -> None:
    if not path.exists():
        raise ValueError("The provided PDF file path does not exist.")

    size = path.stat().st_size
    max_bytes = max_pdf_mb * 1024 * 1024

    if size > max_bytes:
        raise ValueError(f"PDF is larger than the configured {max_pdf_mb}MB limit.")

    if size <= 0:
        raise ValueError("PDF file is empty.")


def prepare_pdf_file(
    file_url: str | None,
    file_path: str | None,
    settings: Settings,
    *,
    max_pdf_mb: int | None = None,
) -> tuple[Path, bool]:
    limit_mb = max_pdf_mb or settings.max_pdf_mb

    if file_path:
        path = Path(file_path)
        validate_pdf_file(path, limit_mb)
        return path, False

    if not file_url:
        raise ValueError("Either fileUrl or filePath is required.")

    max_bytes = limit_mb * 1024 * 1024
    downloaded = 0
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = Path(temp_file.name)

    try:
        with temp_file:
            with httpx.Client(
                timeout=settings.request_timeout_seconds,
                follow_redirects=True,
            ) as client:
                with client.stream("GET", file_url) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue

                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise ValueError(f"PDF is larger than the configured {limit_mb}MB limit.")

                        temp_file.write(chunk)

        validate_pdf_file(temp_path, limit_mb)
        return temp_path, True
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def clean_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_line(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\x00", " ")).strip()


def span_is_bold(span: dict[str, Any]) -> bool:
    font_name = str(span.get("font", "")).lower()
    flags = int(span.get("flags", 0) or 0)

    return "bold" in font_name or bool(flags & 16)


def extract_line_from_pymupdf(
    line: dict[str, Any],
    page_number: int,
) -> ExtractedLine | None:
    spans = line.get("spans") or []
    parts: list[str] = []
    sizes: list[float] = []
    bold_count = 0
    font_name: str | None = None

    for span in spans:
        text = clean_line(str(span.get("text", "")))
        if not text:
            continue

        parts.append(text)
        if isinstance(span.get("size"), (float, int)):
            sizes.append(float(span["size"]))
        if span_is_bold(span):
            bold_count += 1
        if not font_name and span.get("font"):
            font_name = str(span["font"])

    text = clean_line(" ".join(parts))
    if not text:
        return None

    return ExtractedLine(
        bbox=tuple(line.get("bbox")) if line.get("bbox") else None,
        font_name=font_name,
        font_size=max(sizes) if sizes else None,
        is_bold=bold_count > 0,
        page_number=page_number,
        text=text,
    )


def validate_page_count(document, page_limit: int) -> int:
    page_count = document.page_count

    if page_count > page_limit:
        raise ValueError(f"PDF has {page_count} pages, above the configured {page_limit} page limit.")

    return page_count


def extract_pymupdf_page(document, page_index: int) -> ExtractedPage:
    page = document.load_page(page_index)
    lines: list[ExtractedLine] = []
    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            extracted_line = extract_line_from_pymupdf(line, page_index + 1)
            if extracted_line:
                lines.append(extracted_line)

    text = clean_text("\n".join(line.text for line in lines))
    if not text:
        text = clean_text(page.get_text("text"))
        lines = [
            ExtractedLine(
                bbox=None,
                font_name=None,
                font_size=None,
                is_bold=False,
                page_number=page_index + 1,
                text=line,
            )
            for line in text.splitlines()
            if clean_line(line)
        ]

    return ExtractedPage(
        extractor="pymupdf",
        lines=lines,
        page_number=page_index + 1,
        text=text,
    )


def extract_pdf_page_batches(
    pdf_path: Path,
    settings: Settings,
    *,
    max_pdf_pages: int | None = None,
    batch_pages: int | None = None,
):
    page_limit = max_pdf_pages or settings.max_pdf_pages
    batch_size = max(1, batch_pages or settings.extraction_batch_pages)
    document = fitz.open(pdf_path)

    try:
        page_count = validate_page_count(document, page_limit)

        for start in range(0, page_count, batch_size):
            end = min(start + batch_size, page_count)
            yield (
                [extract_pymupdf_page(document, page_index) for page_index in range(start, end)],
                page_count,
                start + 1,
                end,
            )
    finally:
        document.close()


def extract_with_pymupdf(
    pdf_path: Path,
    settings: Settings,
    *,
    max_pdf_pages: int | None = None,
) -> tuple[list[ExtractedPage], int]:
    pages: list[ExtractedPage] = []
    page_count = 0

    for batch, page_count, _, _ in extract_pdf_page_batches(
        pdf_path,
        settings,
        max_pdf_pages=max_pdf_pages,
    ):
        pages.extend(batch)

    return pages, page_count


def extract_pdf_pages(
    pdf_path: Path,
    settings: Settings,
    *,
    max_pdf_pages: int | None = None,
) -> tuple[list[ExtractedPage], int]:
    return extract_with_pymupdf(pdf_path, settings, max_pdf_pages=max_pdf_pages)
