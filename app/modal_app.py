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
        "fastembed==0.8.0",
        "onnxruntime==1.27.0",
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


FIRST_PAGE_THUMBNAIL_DPI = 120
FIRST_PAGE_THUMBNAIL_MAX_WIDTH = 900
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_embedding_model = None


def render_first_page_thumbnail(pdf_path) -> str | None:
    """Renders page 1 to a JPEG, base64-encoded for the JSON response. Best
    effort: a thumbnail failure should never fail the whole processing job."""
    import base64

    import fitz

    document = None
    try:
        document = fitz.open(pdf_path)
        if document.page_count == 0:
            return None

        page = document.load_page(0)
        zoom = FIRST_PAGE_THUMBNAIL_DPI / 72
        if page.rect.width > 0:
            zoom = min(zoom, FIRST_PAGE_THUMBNAIL_MAX_WIDTH / page.rect.width)

        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return base64.b64encode(pixmap.tobytes("jpeg", jpg_quality=70)).decode("ascii")
    except Exception:
        return None
    finally:
        if document is not None:
            document.close()


@app.function(cpu=2.0, memory=4096, timeout=300)
def process_pdf(
    *,
    file_url: str,
    include_summaries: bool = True,
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
        thumbnail_base64 = render_first_page_thumbnail(pdf_path)
        pages, page_count = extract_pdf_pages(pdf_path, settings, max_pdf_pages=max_pdf_pages)
        text_chars = sum(len(page.text) for page in pages)

        sections, warnings = detect_sections(pages)
        outline = build_outline(sections)
        chunks = chunk_sections(sections, settings)
        summaries = build_summaries(sections, chunks) if include_summaries else []

        return {
            "sections": [item.section.model_dump(by_alias=True) for item in sections],
            "chunks": [chunk.model_dump(by_alias=True) for chunk in chunks],
            "summaries": [summary.model_dump(by_alias=True) for summary in summaries],
            "outline": outline,
            "pageCount": page_count,
            "textChars": text_chars,
            "thumbnailBase64": thumbnail_base64,
            "warnings": warnings,
        }
    finally:
        if should_cleanup and pdf_path:
            pdf_path.unlink(missing_ok=True)


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        from fastembed import TextEmbedding

        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    return _embedding_model


@app.function(cpu=1.0, memory=2048, timeout=120, scaledown_window=300)
def embed_texts(*, texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return [embedding.tolist() for embedding in model.embed(texts)]
