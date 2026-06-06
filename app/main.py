from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.config import Settings, get_settings
from app.db import mark_document_failed, mark_document_pending, save_document_chunks
from app.extraction import chunk_pages, extract_pdf_pages, read_pdf_bytes
from app.schemas import ProcessRequest, ProcessResponse

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
    settings: Settings = Depends(get_settings),
) -> None:
    expected = f"Bearer {settings.internal_job_secret}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal service credentials.",
        )


@app.post("/process", response_model=ProcessResponse, dependencies=[Depends(require_internal_secret)])
def process_document(payload: ProcessRequest, settings: Settings = Depends(get_settings)):
    if payload.persist and not settings.database_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DATABASE_URL is required when persist is enabled.",
        )

    try:
        if payload.persist and settings.database_url:
            mark_document_pending(settings.database_url, payload.uploaded_document_id)

        pdf_bytes = read_pdf_bytes(
            str(payload.file_url) if payload.file_url else None,
            payload.file_path,
            settings,
        )
        pages, page_count = extract_pdf_pages(pdf_bytes, settings)
        chunks = chunk_pages(pages, settings)

        if not chunks:
            raise ValueError("No extractable text chunks were found in this PDF.")

        if payload.persist and settings.database_url:
            save_document_chunks(
                settings.database_url,
                payload.uploaded_document_id,
                chunks,
                page_count,
            )

        return ProcessResponse(
            uploadedDocumentId=payload.uploaded_document_id,
            status="READY",
            pageCount=page_count,
            chunkCount=len(chunks),
            chunks=chunks,
        )
    except Exception as error:
        error_message = str(error) or "PDF processing failed."
        if payload.persist and settings.database_url:
            mark_document_failed(settings.database_url, payload.uploaded_document_id, error_message)

        return ProcessResponse(
            uploadedDocumentId=payload.uploaded_document_id,
            status="FAILED",
            pageCount=0,
            chunkCount=0,
            chunks=[],
            error=error_message,
        )
