import re
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

import fitz
import httpx
import pdfplumber
import tiktoken

from app.config import Settings
from app.schemas import ProcessedChunk


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    extractor: str


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


def extract_with_pymupdf(pdf_bytes: bytes, settings: Settings) -> tuple[list[ExtractedPage], int]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = document.page_count

    if page_count > settings.max_pdf_pages:
        document.close()
        raise ValueError(f"PDF has {page_count} pages, above the configured {settings.max_pdf_pages} page limit.")

    pages: list[ExtractedPage] = []
    for page_index in range(page_count):
        page = document.load_page(page_index)
        text = clean_text(page.get_text("text"))
        pages.append(
            ExtractedPage(
                page_number=page_index + 1,
                text=text,
                extractor="pymupdf",
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
            pages.append(
                ExtractedPage(
                    page_number=page_index + 1,
                    text=text,
                    extractor="pdfplumber",
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


def chunk_pages(pages: list[ExtractedPage], settings: Settings) -> list[ProcessedChunk]:
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks: list[ProcessedChunk] = []
    current_parts: list[str] = []
    current_pages: list[int] = []
    current_tokens: list[int] = []
    last_overlap_tokens: list[int] = []

    for page in pages:
        if not page.text:
            continue

        page_text = f"Page {page.page_number}\n{page.text}"
        page_tokens = encoding.encode(page_text)

        if current_tokens and len(current_tokens) + len(page_tokens) > settings.max_chunk_tokens:
            chunk_text = encoding.decode(current_tokens).strip()
            chunks.append(
                ProcessedChunk(
                    chunkIndex=len(chunks),
                    pageStart=min(current_pages) if current_pages else None,
                    pageEnd=max(current_pages) if current_pages else None,
                    text=chunk_text,
                    tokenCount=len(current_tokens),
                    metadata={"extractor": page.extractor},
                )
            )
            last_overlap_tokens = current_tokens[-settings.chunk_overlap_tokens :] if settings.chunk_overlap_tokens > 0 else []
            current_tokens = list(last_overlap_tokens)
            overlap_text = encoding.decode(last_overlap_tokens).strip()
            current_parts = [overlap_text] if overlap_text else []
            current_pages = current_pages[-1:] if current_pages else []

        current_parts.append(page_text)
        current_pages.append(page.page_number)
        current_tokens.extend(page_tokens)

        while len(current_tokens) > settings.max_chunk_tokens:
            slice_tokens = current_tokens[: settings.max_chunk_tokens]
            chunk_text = encoding.decode(slice_tokens).strip()
            chunks.append(
                ProcessedChunk(
                    chunkIndex=len(chunks),
                    pageStart=min(current_pages) if current_pages else page.page_number,
                    pageEnd=max(current_pages) if current_pages else page.page_number,
                    text=chunk_text,
                    tokenCount=len(slice_tokens),
                    metadata={"extractor": page.extractor},
                )
            )
            overlap = slice_tokens[-settings.chunk_overlap_tokens :] if settings.chunk_overlap_tokens > 0 else []
            current_tokens = overlap + current_tokens[settings.max_chunk_tokens :]
            current_parts = [encoding.decode(current_tokens).strip()]

    if current_tokens:
        chunk_text = encoding.decode(current_tokens).strip()
        if chunk_text:
            chunks.append(
                ProcessedChunk(
                    chunkIndex=len(chunks),
                    pageStart=min(current_pages) if current_pages else None,
                    pageEnd=max(current_pages) if current_pages else None,
                    text=chunk_text,
                    tokenCount=len(current_tokens),
                    metadata={"extractor": "mixed"},
                )
            )

    return chunks
