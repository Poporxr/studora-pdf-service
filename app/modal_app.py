"""
Runs the CPU/memory-heavy part of PDF processing (download, PyMuPDF
extraction, structure detection, chunking, summaries) on Modal instead of
the shared Render instance. Pure compute only: no database access, no
secrets. main.py calls `process_pdf.remote(...)` and gets back plain JSON,
then does the exact same Postgres writes it always did.

Deploy with: modal deploy app/modal_app.py
"""

import modal

MAX_EXTRACTION_BATCH_PAGES = 30
MARKITDOWN_CONVERTER_VERSION = "markitdown-v2-text-fast-path"

extraction_image = (
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

embedding_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "fastembed==0.8.0",
        "onnxruntime==1.27.0",
    )
)

markitdown_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "httpx==0.28.1",
        "markitdown[pdf,docx,pptx,xlsx]==0.1.7",
        "regex==2026.5.9",
    )
)

app = modal.App("studora-pdf-processor")


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
PREVIEW_PAGE_DPI = 132
PREVIEW_PAGE_MAX_WIDTH = 1100
PREVIEW_JPEG_QUALITY = 74
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


def render_page_jpeg(page) -> tuple[bytes, int, int]:
    import fitz

    zoom = PREVIEW_PAGE_DPI / 72
    if page.rect.width > 0:
        zoom = min(zoom, PREVIEW_PAGE_MAX_WIDTH / page.rect.width)

    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pixmap.tobytes("jpeg", jpg_quality=PREVIEW_JPEG_QUALITY), pixmap.width, pixmap.height


@app.function(cpu=4.0, memory=8192, timeout=300, image=extraction_image)
def process_pdf(
    *,
    file_url: str,
    include_summaries: bool = True,
    include_thumbnail: bool = True,
    max_pdf_mb: int,
    max_pdf_pages: int,
    extraction_batch_pages: int,
    max_chunk_tokens: int,
    chunk_overlap_tokens: int,
) -> dict:
    from time import perf_counter

    from app.chunking import chunk_sections
    from app.extraction import extract_pdf_page_batches, prepare_pdf_file
    from app.structure import build_outline, detect_sections
    from app.summaries import build_summaries

    settings = _ExtractionSettings(
        max_pdf_mb=max_pdf_mb,
        max_pdf_pages=max_pdf_pages,
        extraction_batch_pages=max(1, min(extraction_batch_pages, MAX_EXTRACTION_BATCH_PAGES)),
        max_chunk_tokens=max_chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )

    pdf_path = None
    should_cleanup = False
    started_at = perf_counter()
    timings: dict[str, object] = {}

    def mark(phase: str, phase_started_at: float) -> None:
        timings[phase] = round((perf_counter() - phase_started_at) * 1000)

    try:
        phase_started_at = perf_counter()
        pdf_path, should_cleanup = prepare_pdf_file(
            file_url,
            None,
            settings,
            max_pdf_mb=max_pdf_mb,
        )
        mark("downloadMs", phase_started_at)

        phase_started_at = perf_counter()
        thumbnail_base64 = render_first_page_thumbnail(pdf_path) if include_thumbnail else None
        mark("thumbnailMs", phase_started_at)

        phase_started_at = perf_counter()
        pages = []
        page_count = 0
        batch_timings = []
        for batch, page_count, start_page, end_page in extract_pdf_page_batches(
            pdf_path,
            settings,
            max_pdf_pages=max_pdf_pages,
            batch_pages=settings.extraction_batch_pages,
        ):
            pages.extend(batch)
            batch_timings.append(
                {
                    "endPage": end_page,
                    "pageCount": len(batch),
                    "startPage": start_page,
                }
            )
        text_chars = sum(len(page.text) for page in pages)
        mark("extractMs", phase_started_at)
        timings["extractBatchCount"] = len(batch_timings)
        timings["extractMaxBatchPages"] = settings.extraction_batch_pages
        timings["extractBatches"] = batch_timings

        phase_started_at = perf_counter()
        sections, warnings = detect_sections(pages)
        outline = build_outline(sections)
        mark("structureMs", phase_started_at)

        phase_started_at = perf_counter()
        chunks = chunk_sections(sections, settings)
        mark("chunkMs", phase_started_at)

        phase_started_at = perf_counter()
        summaries = build_summaries(sections, chunks) if include_summaries else []
        mark("summaryMs", phase_started_at)
        timings["totalMs"] = round((perf_counter() - started_at) * 1000)

        return {
            "sections": [item.section.model_dump(by_alias=True) for item in sections],
            "chunks": [chunk.model_dump(by_alias=True) for chunk in chunks],
            "summaries": [summary.model_dump(by_alias=True) for summary in summaries],
            "outline": outline,
            "pageCount": page_count,
            "textChars": text_chars,
            "thumbnailBase64": thumbnail_base64,
            "timings": timings,
            "warnings": warnings,
        }
    finally:
        if should_cleanup and pdf_path:
            pdf_path.unlink(missing_ok=True)


@app.function(cpu=2.0, memory=4096, timeout=120, image=extraction_image)
def render_pdf_thumbnail(
    *,
    file_url: str,
    max_pdf_mb: int,
    max_pdf_pages: int,
) -> dict:
    from time import perf_counter

    from app.extraction import prepare_pdf_file

    settings = _ExtractionSettings(
        max_pdf_mb=max_pdf_mb,
        max_pdf_pages=max_pdf_pages,
        extraction_batch_pages=1,
        max_chunk_tokens=800,
        chunk_overlap_tokens=120,
    )

    pdf_path = None
    should_cleanup = False
    started_at = perf_counter()

    try:
        pdf_path, should_cleanup = prepare_pdf_file(
            file_url,
            None,
            settings,
            max_pdf_mb=max_pdf_mb,
        )
        thumbnail_base64 = render_first_page_thumbnail(pdf_path)

        return {
            "thumbnailBase64": thumbnail_base64,
            "timings": {
                "totalMs": round((perf_counter() - started_at) * 1000),
            },
        }
    finally:
        if should_cleanup and pdf_path:
            pdf_path.unlink(missing_ok=True)


@app.function(cpu=4.0, memory=4096, timeout=120, image=extraction_image)
def render_pdf_previews(
    *,
    file_url: str,
    upload_targets: list[dict],
    max_pages: int,
    max_pdf_mb: int,
    max_pdf_pages: int,
) -> dict:
    from time import perf_counter

    import fitz
    import httpx

    from app.extraction import prepare_pdf_file

    settings = _ExtractionSettings(
        max_pdf_mb=max_pdf_mb,
        max_pdf_pages=max_pdf_pages,
        extraction_batch_pages=1,
        max_chunk_tokens=800,
        chunk_overlap_tokens=120,
    )

    pdf_path = None
    should_cleanup = False
    started_at = perf_counter()
    pages: list[dict] = []
    warnings: list[str] = []

    try:
        pdf_path, should_cleanup = prepare_pdf_file(
            file_url,
            None,
            settings,
            max_pdf_mb=max_pdf_mb,
        )

        targets_by_page = {
            int(target["pageNumber"]): target
            for target in upload_targets[: max(0, max_pages)]
            if int(target.get("pageNumber", 0) or 0) > 0
        }
        if not targets_by_page:
            return {"pages": [], "warnings": ["No preview upload targets were provided."], "timings": {"totalMs": 0}}

        document = fitz.open(pdf_path)
        try:
            page_limit = min(document.page_count, max_pages, max(targets_by_page))
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                for page_number in range(1, page_limit + 1):
                    target = targets_by_page.get(page_number)
                    if not target:
                        continue

                    page = document.load_page(page_number - 1)
                    image_bytes, width, height = render_page_jpeg(page)
                    response = client.put(
                        str(target["uploadUrl"]),
                        content=image_bytes,
                        headers={"Content-Type": "image/jpeg"},
                    )
                    response.raise_for_status()
                    pages.append(
                        {
                            "pageNumber": page_number,
                            "imageKey": target["imageKey"],
                            "imageUrl": target["imageUrl"],
                            "width": width,
                            "height": height,
                            "blurhash": None,
                        }
                    )
        finally:
            document.close()

        return {
            "pages": pages,
            "warnings": warnings,
            "timings": {
                "totalMs": round((perf_counter() - started_at) * 1000),
            },
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


@app.function(cpu=2.0, memory=4096, timeout=120, scaledown_window=300, image=embedding_image)
def embed_texts(*, texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return [embedding.tolist() for embedding in model.embed(texts)]


@app.function(cpu=4.0, memory=8192, timeout=300, image=markitdown_image)
def convert_markitdown_v2(
    *,
    file_url: str,
    file_suffix: str,
    max_file_mb: int,
    max_chunk_tokens: int,
    chunk_overlap_tokens: int,
) -> dict:
    import hashlib
    import re
    import tempfile
    from pathlib import Path
    from time import perf_counter
    from uuid import uuid4

    import httpx
    from markitdown import MarkItDown

    allowed_suffixes = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".txt", ".md", ".csv", ".json", ".xml", ".epub"}
    normalized_suffix = file_suffix.lower() if file_suffix.startswith(".") else f".{file_suffix.lower()}"
    if normalized_suffix not in allowed_suffixes:
        raise ValueError(f"Unsupported MarkItDown file type: {normalized_suffix}")

    max_bytes = max_file_mb * 1024 * 1024
    downloaded = 0
    timings: dict[str, int] = {}
    started_at = perf_counter()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=normalized_suffix)
    temp_path = Path(temp_file.name)

    def mark(phase: str, phase_started_at: float) -> None:
        timings[phase] = round((perf_counter() - phase_started_at) * 1000)

    def clean_markdown(text: str) -> str:
        cleaned = text.replace("\x00", " ")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def split_markdown_chunks(text: str) -> list[dict]:
        words = re.findall(r"\S+", text)
        chunk_words = max(80, max_chunk_tokens)
        overlap_words = max(0, min(chunk_overlap_tokens, chunk_words // 3))
        chunks = []
        index = 0
        previous_id = None

        while index < len(words):
            chunk_id = str(uuid4())
            end = min(index + chunk_words, len(words))
            chunk_text = " ".join(words[index:end]).strip()
            if not chunk_text:
                break

            chunk = {
                "id": chunk_id,
                "chunkIndex": len(chunks),
                "pageStart": 1,
                "pageEnd": 1,
                "sectionId": section_id,
                "text": chunk_text,
                "tokenCount": len(chunk_text.split()),
                "contentPreview": re.sub(r"\s+", " ", chunk_text).strip()[:240],
                "previousChunkId": previous_id,
                "nextChunkId": None,
                "chunkHash": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                "heading": "Converted document",
                "headingPath": ["Converted document"],
                "sectionNumber": None,
                "metadata": {
                    "converter": "markitdown",
                    "converterVersion": MARKITDOWN_CONVERTER_VERSION,
                    "fileSuffix": normalized_suffix,
                    "source": "markitdown",
                },
            }
            if chunks:
                chunks[-1]["nextChunkId"] = chunk_id
            chunks.append(chunk)
            previous_id = chunk_id

            if end >= len(words):
                break
            index = max(end - overlap_words, index + 1)

        return chunks

    try:
        phase_started_at = perf_counter()
        with temp_file:
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                with client.stream("GET", file_url) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise ValueError(f"File is larger than the configured {max_file_mb}MB limit.")
                        temp_file.write(chunk)
        mark("downloadMs", phase_started_at)

        phase_started_at = perf_counter()
        if normalized_suffix in {".txt", ".md", ".csv", ".json", ".xml"}:
            markdown = clean_markdown(temp_path.read_bytes().decode("utf-8", errors="replace"))
        else:
            markdown = clean_markdown(MarkItDown(enable_plugins=False).convert(str(temp_path)).text_content)
        mark("convertMs", phase_started_at)

        if not markdown:
            raise ValueError("MarkItDown did not extract readable text from this file.")

        section_id = str(uuid4())
        section = {
            "id": section_id,
            "parentSectionId": None,
            "level": 1,
            "title": "Converted document",
            "headingPath": ["Converted document"],
            "pageStart": 1,
            "pageEnd": 1,
            "sortOrder": 0,
            "confidence": 0.5,
            "metadata": {
                "converter": "markitdown",
                "converterVersion": MARKITDOWN_CONVERTER_VERSION,
                "fileSuffix": normalized_suffix,
            },
        }

        phase_started_at = perf_counter()
        chunks = split_markdown_chunks(markdown)
        mark("chunkMs", phase_started_at)
        timings["totalMs"] = round((perf_counter() - started_at) * 1000)

        return {
            "sections": [section],
            "chunks": chunks,
            "summaries": [],
            "outline": [
                {
                    "id": section["id"],
                    "level": section["level"],
                    "title": section["title"],
                    "headingPath": section["headingPath"],
                    "pageStart": section["pageStart"],
                    "pageEnd": section["pageEnd"],
                    "confidence": section["confidence"],
                }
            ],
            "pageCount": 1,
            "textChars": len(markdown),
            "thumbnailBase64": None,
            "timings": timings,
            "warnings": [f"Converted with {MARKITDOWN_CONVERTER_VERSION} experimental path."],
        }
    finally:
        temp_path.unlink(missing_ok=True)
