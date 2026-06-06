from collections.abc import Sequence
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import certifi
import psycopg
from psycopg.types.json import Jsonb

from app.schemas import ProcessedChunk


def normalize_database_url(database_url: str) -> str:
    parts = urlsplit(database_url.strip().strip("'\""))
    query = dict(parse_qsl(parts.query, keep_blank_values=True))

    if query.get("sslmode") == "verify-full":
        query["sslrootcert"] = certifi.where()

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query),
            parts.fragment,
        )
    )


def connect(database_url: str):
    return psycopg.connect(normalize_database_url(database_url), connect_timeout=10)


def mark_document_pending(database_url: str, uploaded_document_id: str) -> None:
    with connect(database_url) as connection:
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
            if cursor.rowcount != 1:
                raise ValueError("Uploaded document was not found in Studora.")


def save_document_chunks(
    database_url: str,
    uploaded_document_id: str,
    chunks: Sequence[ProcessedChunk],
    page_count: int,
) -> None:
    now = datetime.now(UTC)

    with connect(database_url) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM "UploadedDocumentChunk"
                    WHERE "uploadedDocumentId" = %s::uuid
                    """,
                    (uploaded_document_id,),
                )

                cursor.executemany(
                    """
                    INSERT INTO "UploadedDocumentChunk"
                        ("id", "uploadedDocumentId", "chunkIndex", "pageStart", "pageEnd", "text", "tokenCount", "metadata", "createdAt")
                    VALUES
                        (%s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    [
                        (
                            str(uuid4()),
                            uploaded_document_id,
                            chunk.chunk_index,
                            chunk.page_start,
                            chunk.page_end,
                            chunk.text,
                            chunk.token_count,
                            Jsonb(chunk.metadata),
                            now,
                        )
                        for chunk in chunks
                    ],
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
                if cursor.rowcount != 1:
                    raise ValueError("Uploaded document could not be marked as ready.")


def mark_document_failed(database_url: str, uploaded_document_id: str, error: str) -> None:
    with connect(database_url) as connection:
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
            if cursor.rowcount != 1:
                raise ValueError("Uploaded document could not be marked as failed.")
