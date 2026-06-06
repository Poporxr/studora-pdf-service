from collections.abc import Sequence
from datetime import UTC, datetime

import psycopg
from psycopg.types.json import Jsonb

from app.schemas import ProcessedChunk


def mark_document_pending(database_url: str, uploaded_document_id: str) -> None:
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE "UploadedDocument"
                SET "processingStatus" = %s::"UploadedDocumentProcessingStatus",
                    "processingError" = NULL
                WHERE "id" = %s::uuid
                """,
                ("PENDING", uploaded_document_id),
            )


def save_document_chunks(
    database_url: str,
    uploaded_document_id: str,
    chunks: Sequence[ProcessedChunk],
    page_count: int,
) -> None:
    now = datetime.now(UTC)

    with psycopg.connect(database_url) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM "UploadedDocumentChunk"
                    WHERE "uploadedDocumentId" = %s::uuid
                    """,
                    (uploaded_document_id,),
                )

                for chunk in chunks:
                    cursor.execute(
                        """
                        INSERT INTO "UploadedDocumentChunk"
                            ("uploadedDocumentId", "chunkIndex", "pageStart", "pageEnd", "text", "tokenCount", "metadata", "createdAt")
                        VALUES
                            (%s::uuid, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        """,
                        (
                            uploaded_document_id,
                            chunk.chunk_index,
                            chunk.page_start,
                            chunk.page_end,
                            chunk.text,
                            chunk.token_count,
                            Jsonb(chunk.metadata),
                            now,
                        ),
                    )

                cursor.execute(
                    """
                    UPDATE "UploadedDocument"
                    SET "processingStatus" = %s::"UploadedDocumentProcessingStatus",
                        "processingError" = NULL,
                        "processedAt" = %s,
                        "metadata" = COALESCE("metadata", '{}'::jsonb) || %s::jsonb
                    WHERE "id" = %s::uuid
                    """,
                    (
                        "READY",
                        now,
                        Jsonb(
                            {
                                "pdfProcessing": {
                                    "pageCount": page_count,
                                    "chunkCount": len(chunks),
                                    "processedBy": "studora-pdf-service",
                                }
                            }
                        ),
                        uploaded_document_id,
                    ),
                )


def mark_document_failed(database_url: str, uploaded_document_id: str, error: str) -> None:
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE "UploadedDocument"
                SET "processingStatus" = %s::"UploadedDocumentProcessingStatus",
                    "processingError" = %s,
                    "processedAt" = NULL
                WHERE "id" = %s::uuid
                """,
                ("FAILED", error[:1000], uploaded_document_id),
            )
