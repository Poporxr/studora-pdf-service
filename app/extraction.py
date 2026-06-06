from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Any

import fitz
import httpx
import pdfplumber

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


def read_pdf_bytes(file_url: str | None, file_path: str | None, settings: Settings) -> bytes:
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise ValueError("The provided PDF file path does not exist.")
        data = path.read_bytes()
    elif file_url:
        with httpx.Client(timeout=settings.request_timeout_seconds, follow_redirects=True) as client:
            response = client.get(file_url)
            response.raise_for_status()
            data = response.content
    else:
        raise ValueError("Either fileUrl or filePath is required.")

    max_bytes = settings.max_pdf_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise ValueError(f"PDF is larger than the configured {settings.max_pdf_mb}MB limit.")

    if not data:
        raise ValueError("PDF file is empty.")

    return data


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


def extract_with_pymupdf(pdf_bytes: bytes, settings: Settings) -> tuple[list[ExtractedPage], int]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = document.page_count

    if page_count > settings.max_pdf_pages:
        document.close()
        raise ValueError(f"PDF has {page_count} pages, above the configured {settings.max_pdf_pages} page limit.")

    pages: list[ExtractedPage] = []
    for page_index in range(page_count):
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

        pages.append(
            ExtractedPage(
                extractor="pymupdf",
                lines=lines,
                page_number=page_index + 1,
                text=text,
            )
        )

    document.close()
    return pages, page_count


def extract_with_pdfplumber(pdf_bytes: bytes, settings: Settings) -> tuple[list[ExtractedPage], int]:
    pages: list[ExtractedPage] = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        if page_count > settings.max_pdf_pages:
            raise ValueError(f"PDF has {page_count} pages, above the configured {settings.max_pdf_pages} page limit.")

        for page_index, page in enumerate(pdf.pages):
            text = clean_text(page.extract_text() or "")
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
            pages.append(
                ExtractedPage(
                    extractor="pdfplumber",
                    lines=lines,
                    page_number=page_index + 1,
                    text=text,
                )
            )

    return pages, page_count


def extract_pdf_pages(pdf_bytes: bytes, settings: Settings) -> tuple[list[ExtractedPage], int]:
    pages, page_count = extract_with_pymupdf(pdf_bytes, settings)
    extracted_chars = sum(len(page.text) for page in pages)

    if extracted_chars >= 500 or page_count <= 2:
        return pages, page_count

    fallback_pages, fallback_page_count = extract_with_pdfplumber(pdf_bytes, settings)
    fallback_chars = sum(len(page.text) for page in fallback_pages)

    if fallback_chars > extracted_chars:
        return fallback_pages, fallback_page_count

    return pages, page_count
