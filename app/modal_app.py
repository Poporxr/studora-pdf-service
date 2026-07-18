"""
Runs the CPU/memory-heavy part of PDF processing (download, PyMuPDF
extraction, structure detection, chunking, summaries) on Modal instead of
the shared Render instance. Pure compute only: no database access, no
secrets. main.py calls `process_pdf.remote(...)` and gets back plain JSON,
then does the exact same Postgres writes it always did.

Deploy with: modal deploy app/modal_app.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "pymupdf==1.27.2.3",
        "httpx==0.28.1",
        "tiktoken==0.13.0",
        "regex==2026.5.9",
        "pydantic==2.13.4",
        "pydantic-settings==2.14.1",
    )
    .add_local_python_source("app")
)

app = modal.App("studora-pdf-processor", image=image)


class _ExtractionSettings:
    """Duck-types the subset of app.config.Settings that extraction/chunking read."""

    def __init__(
        self,
        *,
        max_pdf_mb: int,
        max_pdf_pages: int,
        extraction_batch_pages: int,
        max_chunk_tokens: int,
        chunk_overlap_tokens: int,
    ) -> None:
        self.max_pdf_mb = max_pdf_mb
        self.max_pdf_pages = max_pdf_pages
        self.extraction_batch_pages = extraction_batch_pages
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.request_timeout_seconds = 60


@app.function(cpu=1.0, memory=2048, timeout=180)
def process_pdf(
    *,
    file_url: str,
    max_pdf_mb: int,
    max_pdf_pages: int,
    extraction_batch_pages: int,
    max_chunk_tokens: int,
    chunk_overlap_tokens: int,
) -> dict:
    from app.chunking import chunk_sections
    from app.extraction import extract_pdf_pages, prepare_pdf_file
    from app.structure import build_outline, detect_sections
    from app.summaries import build_summaries

    settings = _ExtractionSettings(
        max_pdf_mb=max_pdf_mb,
        max_pdf_pages=max_pdf_pages,
        extraction_batch_pages=extraction_batch_pages,
        max_chunk_tokens=max_chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )

    pdf_path = None
    should_cleanup = False

    try:
        pdf_path, should_cleanup = prepare_pdf_file(
            file_url,
            None,
            settings,
            max_pdf_mb=max_pdf_mb,
        )
        pages, page_count = extract_pdf_pages(pdf_path, settings, max_pdf_pages=max_pdf_pages)
        text_chars = sum(len(page.text) for page in pages)

        sections, warnings = detect_sections(pages)
        outline = build_outline(sections)
        chunks = chunk_sections(sections, settings)
        summaries = build_summaries(sections, chunks)

        return {
            "sections": [item.section.model_dump(by_alias=True) for item in sections],
            "chunks": [chunk.model_dump(by_alias=True) for chunk in chunks],
            "summaries": [summary.model_dump(by_alias=True) for summary in summaries],
            "outline": outline,
            "pageCount": page_count,
            "textChars": text_chars,
            "warnings": warnings,
        }
    finally:
        if should_cleanup and pdf_path:
            pdf_path.unlink(missing_ok=True)
