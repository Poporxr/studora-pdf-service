from collections.abc import Sequence
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import certifi
import psycopg
from psycopg.types.json import Jsonb

from app.schemas import ProcessedChunk, ProcessedSection, ProcessedSummary


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
    sections: Sequence[ProcessedSection] | None = None,
    summaries: Sequence[ProcessedSummary] | None = None,
    status: str = "READY",
    warnings: Sequence[str] | None = None,
    outline: Sequence[dict[str, object]] | None = None,
) -> None:
    now = datetime.now(UTC)
    sections = sections or []
    summaries = summaries or []
    warnings = warnings or []
    outline = outline or []

    with connect(database_url) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM "UploadedDocumentSummary"
                    WHERE "uploadedDocumentId" = %s::uuid
                    """,
                    (uploaded_document_id,),
                )
                cursor.execute(
                    """
                    DELETE FROM "UploadedDocumentChunk"
                    WHERE "uploadedDocumentId" = %s::uuid
                    """,
                    (uploaded_document_id,),
                )
                cursor.execute(
                    """
                    DELETE FROM "UploadedDocumentSection"
                    WHERE "uploadedDocumentId" = %s::uuid
                    """,
                    (uploaded_document_id,),
                )

                if sections:
                    cursor.executemany(
                        """
                        INSERT INTO "UploadedDocumentSection"
                            ("id", "uploadedDocumentId", "parentSectionId", "level", "title", "headingPath", "pageStart", "pageEnd", "sortOrder", "confidence", "metadata", "createdAt")
                        VALUES
                            (%s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        """,
                        [
                            (
                                section.id,
                                uploaded_document_id,
                                section.parent_section_id,
                                section.level,
                                section.title,
                                section.heading_path,
                                section.page_start,
                                section.page_end,
                                section.sort_order,
                                section.confidence,
                                Jsonb(section.metadata),
                                now,
                            )
                            for section in sections
                        ],
                    )

                cursor.executemany(
                    """
                    INSERT INTO "UploadedDocumentChunk"
                        ("id", "uploadedDocumentId", "sectionId", "chunkIndex", "pageStart", "pageEnd", "text", "tokenCount", "contentPreview", "previousChunkId", "nextChunkId", "metadata", "createdAt")
                    VALUES
                        (%s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s::uuid, %s::uuid, %s::jsonb, %s)
                    """,
                    [
                        (
                            chunk.id or str(uuid4()),
                            uploaded_document_id,
                            chunk.section_id,
                            chunk.chunk_index,
                            chunk.page_start,
                            chunk.page_end,
                            chunk.text,
                            chunk.token_count,
                            chunk.content_preview,
                            chunk.previous_chunk_id,
                            chunk.next_chunk_id,
                            Jsonb(chunk.metadata),
                            now,
                        )
                        for chunk in chunks
                        ],
                    )

                if summaries:
                    cursor.executemany(
                        """
                        INSERT INTO "UploadedDocumentSummary"
                            ("id", "uploadedDocumentId", "sectionId", "kind", "title", "summary", "keyPoints", "tokenCount", "model", "sourceVersion", "createdAt", "updatedAt")
                        VALUES
                            (%s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                summary.id,
                                uploaded_document_id,
                                summary.section_id,
                                summary.kind,
                                summary.title,
                                summary.summary,
                                summary.key_points,
                                summary.token_count,
                                summary.model,
                                summary.source_version,
                                now,
                                now,
                            )
                            for summary in summaries
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
                        status,
                        now,
                        Jsonb(
                            {
                                "pdfProcessing": {
                                    "warnings": list(warnings),
                                    "outline": list(outline),
                                    "pageCount": page_count,
                                    "chunkCount": len(chunks),
                                    "sectionCount": len(sections),
                                    "summaryCount": len(summaries),
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
