import gc
import logging
from contextlib import nullcontext
from time import perf_counter

import modal
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status

from app.chunking import chunk_sections
from app.config import Settings, get_settings
from app.db import (
    checkout_connection,
    close_pools,
    finish_document_processing,
    insert_document_chunks,
    insert_document_sections,
    insert_document_summaries,
    mark_document_failed,
    mark_document_pending,
    release_connection,
    reset_document_processing_content,
    save_document_chunks,
    try_lock_document,
)
from app.extraction import extract_pdf_pages, prepare_uploaded_pdf_file
from app.schemas import (
    EmbedRequest,
    EmbedResponse,
    ProcessedChunk,
    ProcessedSection,
    ProcessedSummary,
    ProcessRequest,
    ProcessResponse,
)
from app.structure import detect_sections
from app.summaries import build_summaries

logger = logging.getLogger("studora_pdf_service")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
MAX_EMBED_TEXT_CHARS = 4000

app = FastAPI(
    title="Studora PDF Service",
    description="PDF extraction, cleaning and chunking service for Studora",
    version="1.0.0",
)


@app.on_event("shutdown")
def shutdown_db_pools() -> None:
    close_pools()


def build_processing_http_error(error: Exception) -> HTTPException:
    message = str(error) or "PDF processing failed."
    lower_message = message.lower()

    if isinstance(error, HTTPException):
        return error

    if isinstance(error, FileExistsError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=message)

    if isinstance(error, ValueError):
        if "larger than" in lower_message or "above the configured" in lower_message:
            return HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=message)

        if "only pdf" in lower_message or "must be a pdf" in lower_message:
            return HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=message)

        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=message,
    )


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Studora PDF service is running",
    }


def require_internal_secret(
    authorization: str | None = Header(default=None),
    x_internal_job_secret: str | None = Header(default=None, alias="X-Internal-Job-Secret"),
    settings: Settings = Depends(get_settings),
) -> None:
    expected = normalize_secret(settings.internal_job_secret)
    bearer_token = None

    if authorization:
        header_value = authorization.strip()
        if header_value.lower().startswith("bearer "):
            bearer_token = header_value[7:]

    candidates = [
        bearer_token,
        x_internal_job_secret,
    ]

    if any(normalize_secret(candidate) == expected for candidate in candidates):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid internal service credentials.",
    )


def normalize_secret(value: str | None) -> str:
    if not value:
        return ""

    normalized = value.strip()
    if (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in {"'", '"'}
    ):
        normalized = normalized[1:-1].strip()

    return normalized


def get_processing_status(
    *,
    chunk_count: int,
    page_count: int,
    section_count: int,
    text_chars: int,
    warnings: list[str],
) -> str:
    if text_chars < max(500, page_count * 80):
        return "OCR_REQUIRED"

    if chunk_count == 0:
        return "LOW_CONFIDENCE_EXTRACTION"

    if warnings or section_count <= 1:
        return "READY_WITH_WARNINGS"

    return "READY"


_modal_process_pdf_function: modal.Function | None = None
_modal_embed_texts_function: modal.Function | None = None


def get_modal_process_pdf_function() -> modal.Function:
    global _modal_process_pdf_function

    if _modal_process_pdf_function is None:
        _modal_process_pdf_function = modal.Function.from_name("studora-pdf-processor", "process_pdf")

    return _modal_process_pdf_function


def get_modal_embed_texts_function() -> modal.Function:
    global _modal_embed_texts_function

    if _modal_embed_texts_function is None:
        _modal_embed_texts_function = modal.Function.from_name("studora-pdf-processor", "embed_texts")

    return _modal_embed_texts_function


def call_modal_process_pdf(
    *,
    file_url: str,
    max_pdf_mb: int,
    max_pdf_pages: int,
    settings: Settings,
) -> dict:
    """Runs download + PyMuPDF extraction + structure detection + chunking on
    Modal (its own dedicated container) instead of this shared instance, then
    returns plain JSON-safe dicts for sections/chunks/summaries."""
    return get_modal_process_pdf_function().remote(
        file_url=file_url,
        max_pdf_mb=max_pdf_mb,
        max_pdf_pages=max_pdf_pages,
        extraction_batch_pages=settings.extraction_batch_pages,
        max_chunk_tokens=settings.max_chunk_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )


def call_modal_embed_texts(texts: list[str]) -> list[list[float]]:
    """Runs BGE-small embedding on Modal so small Render instances never load
    the embedding model or ONNX runtime into the web process."""
    return get_modal_embed_texts_function().remote(texts=texts)


def process_pdf_request(
    payload: ProcessRequest,
    settings: Settings,
    *,
    persist: bool,
    max_pdf_mb: int,
    max_pdf_pages: int,
) -> ProcessResponse:
    if persist and not settings.database_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DATABASE_URL is required when persist is enabled.",
        )

    if not payload.file_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="fileUrl is required: PDF processing now runs on Modal, which needs a fetchable URL.",
        )

    started_at = perf_counter()
    lock_connection = None

    try:
        logger.info(
            "pdf processing started uploadedDocumentId=%s source=%s persist=%s",
            payload.uploaded_document_id,
            payload.source,
            persist,
        )

        if persist and settings.database_url:
            lock_connection = checkout_connection(settings.database_url)

        transaction_ctx = lock_connection.transaction() if lock_connection else nullcontext()

        with transaction_ctx:
            if lock_connection:
                if not try_lock_document(lock_connection, payload.uploaded_document_id):
                    raise FileExistsError("Uploaded document is already being processed.")
                mark_document_pending(lock_connection, payload.uploaded_document_id)

            result = call_modal_process_pdf(
                file_url=str(payload.file_url),
                max_pdf_mb=max_pdf_mb,
                max_pdf_pages=max_pdf_pages,
                settings=settings,
            )
            sections = [ProcessedSection(**item) for item in result["sections"]]
            chunks = [ProcessedChunk(**item) for item in result["chunks"]]
            summaries = [ProcessedSummary(**item) for item in result["summaries"]]
            page_count = result["pageCount"]
            text_chars = result["textChars"]
            warnings = list(result["warnings"])

            processing_status = get_processing_status(
                chunk_count=len(chunks),
                page_count=page_count,
                section_count=len(sections),
                text_chars=text_chars,
                warnings=warnings,
            )
            logger.info(
                "pdf structured uploadedDocumentId=%s sectionCount=%s chunkCount=%s summaryCount=%s status=%s warnings=%s elapsedSeconds=%.2f",
                payload.uploaded_document_id,
                len(sections),
                len(chunks),
                len(summaries),
                processing_status,
                len(warnings),
                perf_counter() - started_at,
            )

            if not chunks:
                raise ValueError("No extractable text chunks were found in this PDF.")

            if lock_connection:
                save_document_chunks(
                    lock_connection,
                    payload.uploaded_document_id,
                    chunks,
                    page_count,
                    sections=sections,
                    summaries=summaries,
                    status=processing_status,
                    warnings=warnings,
                    outline=result["outline"],
                )
                logger.info(
                    "pdf chunks saved uploadedDocumentId=%s sectionCount=%s chunkCount=%s summaryCount=%s status=%s elapsedSeconds=%.2f",
                    payload.uploaded_document_id,
                    len(sections),
                    len(chunks),
                    len(summaries),
                    processing_status,
                    perf_counter() - started_at,
                )

            return ProcessResponse(
                uploadedDocumentId=payload.uploaded_document_id,
                status=processing_status,
                pageCount=page_count,
                sectionCount=len(sections),
                chunkCount=len(chunks),
                chunks=[] if persist else chunks,
                sections=[] if persist else sections,
                summaries=[] if persist else summaries,
                warnings=warnings,
            )
    except Exception as error:
        error_message = str(error) or "PDF processing failed."
        
        if not isinstance(error, FileExistsError):
            logger.exception(
                "pdf processing failed uploadedDocumentId=%s error=%s",
                payload.uploaded_document_id,
                error_message,
            )

            if persist and settings.database_url and lock_connection:
                try:
                    with lock_connection.transaction():
                        mark_document_failed(lock_connection, payload.uploaded_document_id, error_message)
                except Exception as update_error:
                    error_message = (
                        f"{error_message} Failed to update document status: {update_error}"
                    )
                    logger.exception(
                        "pdf failure status update failed uploadedDocumentId=%s error=%s",
                        payload.uploaded_document_id,
                        update_error,
                    )

        raise build_processing_http_error(error)
    finally:
        if lock_connection:
            release_connection(settings.database_url, lock_connection)


def process_temporary_pdf_path(
    *,
    payload: ProcessRequest,
    pdf_path,
    settings: Settings,
    started_at: float,
) -> ProcessResponse:
    pages, page_count = extract_pdf_pages(
        pdf_path,
        settings,
        max_pdf_pages=settings.temporary_max_pdf_pages,
    )
    text_chars = sum(len(page.text) for page in pages)
    logger.info(
        "temporary pdf extracted uploadedDocumentId=%s pageCount=%s textChars=%s elapsedSeconds=%.2f",
        payload.uploaded_document_id,
        page_count,
        text_chars,
        perf_counter() - started_at,
    )

    sections, warnings = detect_sections(pages)
    del pages
    gc.collect()
    chunks = chunk_sections(sections, settings)
    processed_sections = [item.section for item in sections]
    summaries = build_summaries(sections, chunks)
    processing_status = get_processing_status(
        chunk_count=len(chunks),
        page_count=page_count,
        section_count=len(processed_sections),
        text_chars=text_chars,
        warnings=warnings,
    )

    if not chunks:
        raise ValueError("No extractable text chunks were found in this PDF.")

    return ProcessResponse(
        uploadedDocumentId=payload.uploaded_document_id,
        status=processing_status,
        pageCount=page_count,
        sectionCount=len(processed_sections),
        chunkCount=len(chunks),
        chunks=chunks,
        sections=processed_sections,
        summaries=summaries,
        warnings=warnings,
    )


def process_resource_progressive(
    payload: ProcessRequest,
    settings: Settings,
    *,
    max_pdf_mb: int,
    max_pdf_pages: int,
) -> ProcessResponse:
    if not settings.database_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DATABASE_URL is required when persist is enabled.",
        )

    if not payload.file_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="fileUrl is required: PDF processing now runs on Modal, which needs a fetchable URL.",
        )

    started_at = perf_counter()
    lock_connection = None

    try:
        logger.info(
            "pdf resource processing started uploadedDocumentId=%s source=%s",
            payload.uploaded_document_id,
            payload.source,
        )
        lock_connection = checkout_connection(settings.database_url)

        with lock_connection.transaction():
            if not try_lock_document(lock_connection, payload.uploaded_document_id):
                raise FileExistsError("Uploaded document is already being processed.")

            mark_document_pending(lock_connection, payload.uploaded_document_id)
            reset_document_processing_content(lock_connection, payload.uploaded_document_id)

            result = call_modal_process_pdf(
                file_url=str(payload.file_url),
                max_pdf_mb=max_pdf_mb,
                max_pdf_pages=max_pdf_pages,
                settings=settings,
            )
            sections = [ProcessedSection(**item) for item in result["sections"]]
            chunks = [ProcessedChunk(**item) for item in result["chunks"]]
            summaries = [ProcessedSummary(**item) for item in result["summaries"]]
            page_count = result["pageCount"]
            text_chars = result["textChars"]
            warnings = list(result["warnings"])
            logger.info(
                "pdf resource processed on modal uploadedDocumentId=%s pageCount=%s sectionCount=%s chunkCount=%s summaryCount=%s elapsedSeconds=%.2f",
                payload.uploaded_document_id,
                page_count,
                len(sections),
                len(chunks),
                len(summaries),
                perf_counter() - started_at,
            )

            processing_status = get_processing_status(
                chunk_count=len(chunks),
                page_count=page_count,
                section_count=len(sections),
                text_chars=text_chars,
                warnings=warnings,
            )
            if not chunks:
                raise ValueError("No extractable text chunks were found in this PDF.")

            insert_document_sections(lock_connection, payload.uploaded_document_id, sections)
            insert_document_chunks(lock_connection, payload.uploaded_document_id, chunks)
            insert_document_summaries(lock_connection, payload.uploaded_document_id, summaries)

            finish_document_processing(
                lock_connection,
                payload.uploaded_document_id,
                chunk_count=len(chunks),
                outline=result["outline"],
                page_count=page_count,
                section_count=len(sections),
                status=processing_status,
                summary_count=len(summaries),
                warnings=warnings,
            )
            logger.info(
                "pdf resource processing finished uploadedDocumentId=%s pageCount=%s sectionCount=%s chunkCount=%s summaryCount=%s status=%s elapsedSeconds=%.2f",
                payload.uploaded_document_id,
                page_count,
                len(sections),
                len(chunks),
                len(summaries),
                processing_status,
                perf_counter() - started_at,
            )

            return ProcessResponse(
                uploadedDocumentId=payload.uploaded_document_id,
                status=processing_status,
                pageCount=page_count,
                sectionCount=len(sections),
                chunkCount=len(chunks),
                chunks=[],
                sections=[],
                summaries=[],
                thumbnailBase64=result.get("thumbnailBase64"),
                warnings=warnings,
            )
    except Exception as error:
        error_message = str(error) or "PDF processing failed."

        if not isinstance(error, FileExistsError):
            logger.exception(
                "pdf resource processing failed uploadedDocumentId=%s error=%s",
                payload.uploaded_document_id,
                error_message,
            )

            if lock_connection:
                try:
                    with lock_connection.transaction():
                        mark_document_failed(lock_connection, payload.uploaded_document_id, error_message)
                except Exception as update_error:
                    error_message = f"{error_message} Failed to update document status: {update_error}"
                    logger.exception(
                        "pdf resource failure status update failed uploadedDocumentId=%s error=%s",
                        payload.uploaded_document_id,
                        update_error,
                    )

        raise build_processing_http_error(error)
    finally:
        if lock_connection:
            release_connection(settings.database_url, lock_connection)


@app.post("/process-resource", response_model=ProcessResponse, dependencies=[Depends(require_internal_secret)])
def process_resource(payload: ProcessRequest, settings: Settings = Depends(get_settings)):
    return process_resource_progressive(
        payload,
        settings,
        max_pdf_mb=settings.max_pdf_mb,
        max_pdf_pages=settings.max_pdf_pages,
    )


@app.post("/extract-temporary", response_model=ProcessResponse, dependencies=[Depends(require_internal_secret)])
def extract_temporary(payload: ProcessRequest, settings: Settings = Depends(get_settings)):
    return process_pdf_request(
        payload,
        settings,
        persist=False,
        max_pdf_mb=settings.temporary_max_pdf_mb,
        max_pdf_pages=settings.temporary_max_pdf_pages,
    )


@app.post("/extract-temporary/upload", response_model=ProcessResponse, dependencies=[Depends(require_internal_secret)])
async def extract_temporary_upload(
    file: UploadFile = File(...),
    uploaded_document_id: str = Form(default="temporary-upload", alias="uploadedDocumentId"),
    source: str = Form(default="temporary-upload"),
    settings: Settings = Depends(get_settings),
):
    started_at = perf_counter()
    pdf_path = None

    try:
        payload = ProcessRequest(
            uploadedDocumentId=uploaded_document_id,
            source=source,
            persist=False,
        )
        pdf_path = await prepare_uploaded_pdf_file(
            file,
            settings,
            max_pdf_mb=settings.temporary_max_pdf_mb,
        )
        return process_temporary_pdf_path(
            payload=payload,
            pdf_path=pdf_path,
            settings=settings,
            started_at=started_at,
        )
    except Exception as error:
        error_message = str(error) or "Temporary PDF extraction failed."
        logger.exception(
            "temporary pdf upload extraction failed uploadedDocumentId=%s error=%s",
            uploaded_document_id,
            error_message,
        )
        raise build_processing_http_error(error)
    finally:
        if pdf_path:
            pdf_path.unlink(missing_ok=True)


@app.post("/process", response_model=ProcessResponse, dependencies=[Depends(require_internal_secret)])
def process_document(payload: ProcessRequest, settings: Settings = Depends(get_settings)):
    return process_pdf_request(
        payload,
        settings,
        persist=payload.persist,
        max_pdf_mb=settings.max_pdf_mb if payload.persist else settings.temporary_max_pdf_mb,
        max_pdf_pages=settings.max_pdf_pages if payload.persist else settings.temporary_max_pdf_pages,
    )


@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_internal_secret)])
def embed(payload: EmbedRequest):
    texts = [text[:MAX_EMBED_TEXT_CHARS] for text in payload.texts]
    embeddings = call_modal_embed_texts(texts)

    return EmbedResponse(
        embeddings=embeddings,
        dimensions=EMBEDDING_DIMENSIONS,
        model=EMBEDDING_MODEL_NAME,
    )
