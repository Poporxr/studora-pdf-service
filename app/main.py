import logging
from time import perf_counter

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.chunking import chunk_sections
from app.config import Settings, get_settings
from app.db import mark_document_failed, mark_document_pending, save_document_chunks
from app.extraction import extract_pdf_pages, read_pdf_bytes
from app.schemas import ProcessRequest, ProcessResponse
from app.structure import build_outline, detect_sections

logger = logging.getLogger("studora_pdf_service")

app = FastAPI(
    title="Studora PDF Service",
    description="PDF extraction, cleaning and chunking service for Studora",
    version="1.0.0",
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


@app.post("/process", response_model=ProcessResponse, dependencies=[Depends(require_internal_secret)])
def process_document(payload: ProcessRequest, settings: Settings = Depends(get_settings)):
    if payload.persist and not settings.database_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DATABASE_URL is required when persist is enabled.",
        )

    started_at = perf_counter()

    try:
        logger.info(
            "pdf processing started uploadedDocumentId=%s source=%s persist=%s",
            payload.uploaded_document_id,
            payload.source,
            payload.persist,
        )

        if payload.persist and settings.database_url:
            mark_document_pending(settings.database_url, payload.uploaded_document_id)

        pdf_bytes = read_pdf_bytes(
            str(payload.file_url) if payload.file_url else None,
            payload.file_path,
            settings,
        )
        logger.info(
            "pdf downloaded uploadedDocumentId=%s sizeBytes=%s",
            payload.uploaded_document_id,
            len(pdf_bytes),
        )

        pages, page_count = extract_pdf_pages(pdf_bytes, settings)
        text_chars = sum(len(page.text) for page in pages)
        logger.info(
            "pdf extracted uploadedDocumentId=%s pageCount=%s textChars=%s",
            payload.uploaded_document_id,
            page_count,
            text_chars,
        )

        sections, warnings = detect_sections(pages)
        outline = build_outline(sections)
        chunks = chunk_sections(sections, settings)
        processing_status = get_processing_status(
            chunk_count=len(chunks),
            page_count=page_count,
            section_count=len(sections),
            text_chars=text_chars,
            warnings=warnings,
        )
        logger.info(
            "pdf structured uploadedDocumentId=%s sectionCount=%s chunkCount=%s status=%s warnings=%s elapsedSeconds=%.2f",
            payload.uploaded_document_id,
            len(sections),
            len(chunks),
            processing_status,
            len(warnings),
            perf_counter() - started_at,
        )

        if not chunks:
            raise ValueError("No extractable text chunks were found in this PDF.")

        if payload.persist and settings.database_url:
            save_document_chunks(
                settings.database_url,
                payload.uploaded_document_id,
                chunks,
                page_count,
                sections=sections,
                status=processing_status,
                warnings=warnings,
                outline=outline,
            )
            logger.info(
                "pdf chunks saved uploadedDocumentId=%s sectionCount=%s chunkCount=%s status=%s elapsedSeconds=%.2f",
                payload.uploaded_document_id,
                len(sections),
                len(chunks),
                processing_status,
                perf_counter() - started_at,
            )

        return ProcessResponse(
            uploadedDocumentId=payload.uploaded_document_id,
            status=processing_status,
            pageCount=page_count,
            sectionCount=len(sections),
            chunkCount=len(chunks),
            chunks=chunks,
            sections=[item.section for item in sections],
            warnings=warnings,
        )
    except Exception as error:
        error_message = str(error) or "PDF processing failed."
        logger.exception(
            "pdf processing failed uploadedDocumentId=%s error=%s",
            payload.uploaded_document_id,
            error_message,
        )

        if payload.persist and settings.database_url:
            try:
                mark_document_failed(settings.database_url, payload.uploaded_document_id, error_message)
            except Exception as update_error:
                error_message = (
                    f"{error_message} Failed to update document status: {update_error}"
                )
                logger.exception(
                    "pdf failure status update failed uploadedDocumentId=%s error=%s",
                    payload.uploaded_document_id,
                    update_error,
                )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message,
        )
