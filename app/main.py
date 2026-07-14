import gc
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status

from app.chunking import chunk_sections
from app.config import Settings, get_settings
from app.db import (
    finish_document_processing,
    insert_document_chunks,
    insert_document_sections,
    insert_document_summaries,
    mark_document_failed,
    mark_document_pending,
    reset_document_processing_content,
    save_document_chunks,
    update_document_chunk_next_id,
    update_document_section_page_end,
)
from app.extraction import (
    extract_pdf_page_batches,
    extract_pdf_pages,
    prepare_pdf_file,
    prepare_uploaded_pdf_file,
)
from app.schemas import ProcessedSection, ProcessRequest, ProcessResponse
from app.structure import (
    SectionText,
    build_heading_path,
    build_outline,
    detect_sections,
    get_font_baseline,
    is_toc_entry,
    score_heading,
)
from app.summaries import build_document_summary, build_section_summary, build_summaries

logger = logging.getLogger("studora_pdf_service")

app = FastAPI(
    title="Studora PDF Service",
    description="PDF extraction, cleaning and chunking service for Studora",
    version="1.0.0",
)


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


def extract_pages_for_processing(
    *,
    max_pdf_pages: int,
    payload: ProcessRequest,
    pdf_path,
    persist: bool,
    settings: Settings,
    started_at: float,
):
    if not persist:
        pages, page_count = extract_pdf_pages(
            pdf_path,
            settings,
            max_pdf_pages=max_pdf_pages,
        )
        return pages, page_count

    pages = []
    page_count = 0

    for batch, page_count, page_start, page_end in extract_pdf_page_batches(
        pdf_path,
        settings,
        max_pdf_pages=max_pdf_pages,
    ):
        pages.extend(batch)
        logger.info(
            "pdf extraction batch uploadedDocumentId=%s pages=%s-%s/%s retainedPages=%s elapsedSeconds=%.2f",
            payload.uploaded_document_id,
            page_start,
            page_end,
            page_count,
            len(pages),
            perf_counter() - started_at,
        )

    return pages, page_count


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

    started_at = perf_counter()
    pdf_path = None
    should_cleanup = False

    try:
        logger.info(
            "pdf processing started uploadedDocumentId=%s source=%s persist=%s",
            payload.uploaded_document_id,
            payload.source,
            persist,
        )

        if persist and settings.database_url:
            mark_document_pending(settings.database_url, payload.uploaded_document_id)

        pdf_path, should_cleanup = prepare_pdf_file(
            str(payload.file_url) if payload.file_url else None,
            payload.file_path,
            settings,
            max_pdf_mb=max_pdf_mb,
        )
        logger.info(
            "pdf prepared uploadedDocumentId=%s path=%s",
            payload.uploaded_document_id,
            pdf_path,
        )

        pages, page_count = extract_pages_for_processing(
            max_pdf_pages=max_pdf_pages,
            payload=payload,
            pdf_path=pdf_path,
            persist=persist,
            settings=settings,
            started_at=started_at,
        )

        text_chars = sum(len(page.text) for page in pages)
        logger.info(
            "pdf extracted uploadedDocumentId=%s pageCount=%s textChars=%s",
            payload.uploaded_document_id,
            page_count,
            text_chars,
        )

        sections, warnings = detect_sections(pages)
        del pages
        gc.collect()
        outline = build_outline(sections)
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

        if persist and settings.database_url:
            save_document_chunks(
                settings.database_url,
                payload.uploaded_document_id,
                chunks,
                page_count,
                sections=processed_sections,
                summaries=summaries,
                status=processing_status,
                warnings=warnings,
                outline=outline,
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
            sectionCount=len(processed_sections),
            chunkCount=len(chunks),
            chunks=[] if persist else chunks,
            sections=[] if persist else processed_sections,
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

            if persist and settings.database_url:
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

        raise build_processing_http_error(error)
    finally:
        if should_cleanup and pdf_path:
            pdf_path.unlink(missing_ok=True)


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

    started_at = perf_counter()
    pdf_path = None
    should_cleanup = False

    baseline_font_size = None
    chunk_count = 0
    current: SectionText | None = None
    document_parts: list[str] = []
    inserted_section_ids: set[str] = set()
    last_chunk_id: str | None = None
    outline: list[dict[str, object]] = []
    page_count = 0
    section_count = 0
    section_summaries = []
    stack: list[ProcessedSection] = []
    summarized_section_ids: set[str] = set()
    text_chars = 0
    warnings: list[str] = []

    def insert_section(section: ProcessedSection) -> None:
        nonlocal section_count
        if section.id in inserted_section_ids:
            return

        insert_document_sections(settings.database_url, payload.uploaded_document_id, [section])
        inserted_section_ids.add(section.id)
        section_count += 1

        if len(outline) < 80:
            outline.append(
                {
                    "id": section.id,
                    "level": section.level,
                    "title": section.title,
                    "headingPath": section.heading_path,
                    "pageStart": section.page_start,
                    "pageEnd": section.page_end,
                    "confidence": section.confidence,
                }
            )

    def flush_current() -> None:
        nonlocal chunk_count, current, last_chunk_id

        if current is None or not current.lines:
            return

        update_document_section_page_end(
            settings.database_url,
            current.section.id,
            current.section.page_end,
        )
        chunks = chunk_sections(
            [current],
            settings,
            previous_chunk_id=last_chunk_id,
            start_index=chunk_count,
        )

        if chunks:
            if last_chunk_id and chunks[0].id:
                update_document_chunk_next_id(settings.database_url, last_chunk_id, chunks[0].id)

            insert_document_chunks(settings.database_url, payload.uploaded_document_id, chunks)
            summary, document_part = build_section_summary(current, chunks)
            if document_part:
                document_parts.append(document_part)
            if (
                summary
                and current.section.id not in summarized_section_ids
                and len(section_summaries) < 40
            ):
                section_summaries.append(summary)
                summarized_section_ids.add(current.section.id)

            chunk_count += len(chunks)
            last_chunk_id = chunks[-1].id

        current.lines.clear()

    try:
        logger.info(
            "pdf resource processing started uploadedDocumentId=%s source=%s",
            payload.uploaded_document_id,
            payload.source,
        )
        mark_document_pending(settings.database_url, payload.uploaded_document_id)
        reset_document_processing_content(settings.database_url, payload.uploaded_document_id)

        pdf_path, should_cleanup = prepare_pdf_file(
            str(payload.file_url) if payload.file_url else None,
            payload.file_path,
            settings,
            max_pdf_mb=max_pdf_mb,
        )

        for batch, page_count, page_start, page_end in extract_pdf_page_batches(
            pdf_path,
            settings,
            max_pdf_pages=max_pdf_pages,
        ):
            if baseline_font_size is None:
                baseline_font_size = get_font_baseline(batch)

            for page in batch:
                text_chars += len(page.text)
                for line in page.lines:
                    text = line.text.strip()
                    if not text or is_toc_entry(text):
                        continue

                    score, level = score_heading(line, baseline_font_size)
                    if score >= 4:
                        flush_current()

                        while stack and stack[-1].level >= level:
                            stack.pop()

                        section = ProcessedSection(
                            id=str(uuid4()),
                            parentSectionId=stack[-1].id if stack else None,
                            level=level,
                            title=text[:180],
                            headingPath=build_heading_path(stack, text[:180], level),
                            pageStart=page.page_number,
                            pageEnd=page.page_number,
                            sortOrder=section_count,
                            confidence=min(score / 8, 0.98),
                            metadata={
                                "fontSize": line.font_size,
                                "isBold": line.is_bold,
                                "detectorScore": score,
                            },
                        )
                        insert_section(section)
                        current = SectionText(section=section)
                        stack.append(section)
                        continue

                    if current is None:
                        section = ProcessedSection(
                            id=str(uuid4()),
                            parentSectionId=None,
                            level=1,
                            title="Document start",
                            headingPath=["Document start"],
                            pageStart=page.page_number,
                            pageEnd=page.page_number,
                            sortOrder=section_count,
                            confidence=0.4,
                            metadata={"generatedFallback": True},
                        )
                        insert_section(section)
                        current = SectionText(section=section)
                        stack = [section]

                    current.lines.append(line)
                    current.section.page_end = page.page_number

            flush_current()
            gc.collect()
            logger.info(
                "pdf resource batch saved uploadedDocumentId=%s pages=%s-%s/%s sections=%s chunks=%s textChars=%s elapsedSeconds=%.2f",
                payload.uploaded_document_id,
                page_start,
                page_end,
                page_count,
                section_count,
                chunk_count,
                text_chars,
                perf_counter() - started_at,
            )

        flush_current()

        if section_count <= 1 and text_chars > 2000:
            warnings.append("Document structure confidence is low; sections were streamed with fallback metadata.")
        elif section_count < max(2, page_count // 20):
            warnings.append("Few headings were detected; section metadata may be incomplete.")

        processing_status = get_processing_status(
            chunk_count=chunk_count,
            page_count=page_count,
            section_count=section_count,
            text_chars=text_chars,
            warnings=warnings,
        )
        if chunk_count == 0:
            raise ValueError("No extractable text chunks were found in this PDF.")

        document_summary = build_document_summary(document_parts)
        summaries = [*section_summaries]
        if document_summary:
            summaries.insert(0, document_summary)
        insert_document_summaries(settings.database_url, payload.uploaded_document_id, summaries)

        finish_document_processing(
            settings.database_url,
            payload.uploaded_document_id,
            chunk_count=chunk_count,
            outline=outline,
            page_count=page_count,
            section_count=section_count,
            status=processing_status,
            summary_count=len(summaries),
            warnings=warnings,
        )
        logger.info(
            "pdf resource processing finished uploadedDocumentId=%s pageCount=%s sectionCount=%s chunkCount=%s summaryCount=%s status=%s elapsedSeconds=%.2f",
            payload.uploaded_document_id,
            page_count,
            section_count,
            chunk_count,
            len(summaries),
            processing_status,
            perf_counter() - started_at,
        )

        return ProcessResponse(
            uploadedDocumentId=payload.uploaded_document_id,
            status=processing_status,
            pageCount=page_count,
            sectionCount=section_count,
            chunkCount=chunk_count,
            chunks=[],
            sections=[],
            summaries=[],
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

            try:
                mark_document_failed(settings.database_url, payload.uploaded_document_id, error_message)
            except Exception as update_error:
                error_message = f"{error_message} Failed to update document status: {update_error}"
                logger.exception(
                    "pdf resource failure status update failed uploadedDocumentId=%s error=%s",
                    payload.uploaded_document_id,
                    update_error,
                )

        raise build_processing_http_error(error)
    finally:
        if should_cleanup and pdf_path:
            pdf_path.unlink(missing_ok=True)


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
