from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from app.extraction import ExtractedLine, ExtractedPage, clean_line, clean_text


AZURE_OCR_VERSION = "azure-document-intelligence-read-v1"
DEFAULT_MODEL_ID = "prebuilt-read"
DEFAULT_API_VERSION = "2024-11-30"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_PAGES = 100
OCR_MIN_TOTAL_TEXT_CHARS = 500
OCR_MIN_TEXT_CHARS_PER_PAGE = 80


@dataclass
class AzureOcrResult:
    pages: list[ExtractedPage]
    text_chars: int
    warnings: list[str]


def env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw.strip())
    except ValueError:
        return default

    return value if value > 0 else default


def should_run_azure_ocr(*, chunk_count: int, page_count: int, text_chars: int) -> bool:
    if not env_enabled("AZURE_OCR_ENABLED"):
        return False

    return text_chars < max(OCR_MIN_TOTAL_TEXT_CHARS, page_count * OCR_MIN_TEXT_CHARS_PER_PAGE) or chunk_count == 0


def get_azure_config() -> tuple[str, str, str, str, int, int]:
    endpoint = (os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT") or "").strip().strip("'\"")
    key = (os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY") or "").strip().strip("'\"")
    model_id = (os.getenv("AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID") or DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    api_version = (
        os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION") or DEFAULT_API_VERSION
    ).strip() or DEFAULT_API_VERSION
    timeout_seconds = env_int("AZURE_OCR_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    max_pages = env_int("AZURE_OCR_MAX_PAGES", DEFAULT_MAX_PAGES)

    if not endpoint:
        raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is required when Azure OCR is enabled.")

    if not key:
        raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_KEY is required when Azure OCR is enabled.")

    return endpoint, key, model_id, api_version, timeout_seconds, max_pages


def line_polygon_to_bbox(polygon: Any) -> tuple[float, float, float, float] | None:
    if not polygon:
        return None

    points: list[tuple[float, float]] = []
    for point in polygon:
        x = getattr(point, "x", None)
        y = getattr(point, "y", None)

        if x is None and isinstance(point, dict):
            x = point.get("x")
            y = point.get("y")

        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
            points.append((float(x), float(y)))

    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def strip_ocr_noise(text: str) -> str:
    cleaned = re.sub(r"\s*\[?selection mark[^\n]*", "", text, flags=re.IGNORECASE)
    return clean_line(cleaned)


def azure_page_to_extracted_page(page: Any) -> ExtractedPage:
    page_number = int(getattr(page, "page_number", None) or getattr(page, "pageNumber", None) or 1)
    lines: list[ExtractedLine] = []

    for line in getattr(page, "lines", []) or []:
        text = strip_ocr_noise(str(getattr(line, "content", "") or ""))
        if not text:
            continue

        lines.append(
            ExtractedLine(
                bbox=line_polygon_to_bbox(getattr(line, "polygon", None)),
                font_name=None,
                font_size=None,
                is_bold=False,
                page_number=page_number,
                text=text,
            )
        )

    text = clean_text("\n".join(line.text for line in lines))
    return ExtractedPage(
        extractor=AZURE_OCR_VERSION,
        lines=lines,
        page_number=page_number,
        text=text,
    )


def analyze_pdf_with_azure_ocr(pdf_path: Path) -> AzureOcrResult:
    endpoint, key, model_id, api_version, timeout_seconds, max_pages = get_azure_config()

    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
        api_version=api_version,
    )

    with pdf_path.open("rb") as file:
        poller = client.begin_analyze_document(
            model_id=model_id,
            body=file,
            pages=f"1-{max_pages}",
        )
        result = poller.result(timeout=timeout_seconds)

    pages = [
        azure_page_to_extracted_page(page)
        for page in getattr(result, "pages", []) or []
    ]
    pages = [page for page in pages if page.text]
    text_chars = sum(len(page.text) for page in pages)
    warnings = [f"Recovered sparse PDF text with {AZURE_OCR_VERSION}."]

    if len(pages) >= max_pages:
        warnings.append(f"Azure OCR was capped at {max_pages} pages.")

    if not pages:
        warnings.append("Azure OCR did not return readable text.")

    return AzureOcrResult(
        pages=pages,
        text_chars=text_chars,
        warnings=warnings,
    )
