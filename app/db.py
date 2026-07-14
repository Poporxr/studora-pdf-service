from collections.abc import Sequence
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import certifi
import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.schemas import ProcessedChunk, ProcessedSection, ProcessedSummary

INSERT_BATCH_SIZE = 200

_pools: dict[str, ConnectionPool] = {}


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


def get_pool(database_url: str) -> ConnectionPool:
    conninfo = normalize_database_url(database_url)
    pool = _pools.get(conninfo)
    if pool is None:
        pool = ConnectionPool(
            conninfo,
            min_size=1,
            max_size=10,
            max_idle=300,
            kwargs={"connect_timeout": 10},
            open=True,
        )
        _pools[conninfo] = pool
    return pool


def close_pools() -> None:
    for pool in _pools.values():
        pool.close()
    _pools.clear()


def checkout_connection(database_url: str) -> psycopg.Connection:
    return get_pool(database_url).getconn()


def release_connection(database_url: str, connection: psycopg.Connection) -> None:
    get_pool(database_url).putconn(connection)


def try_lock_document(connection: psycopg.Connection, uploaded_document_id: str) -> bool:
    """Acquire a transaction-scoped advisory lock for this document.

    Must be called on `connection` while it has an open transaction (e.g. inside
    a `with connection.transaction():` block): the lock is released automatically
    when that transaction commits or rolls back. Neon's pooled endpoint can hand
    out a different backend session per transaction, so a *session*-scoped lock
    (pg_try_advisory_lock) would not reliably survive across separate statements —
    only a lock held for the lifetime of a single transaction is safe here.
    """
    import hashlib
    lock_key = int(hashlib.sha256(uploaded_document_id.encode("utf-8")).hexdigest()[:15], 16)

    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_try_advisory_xact_lock(%s)", (lock_key,))
        return cursor.fetchone()[0]


def mark_document_pending(connection: psycopg.Connection, uploaded_document_id: str) -> None:
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
        if cursor.rowcount == 0:
            raise ValueError("Uploaded document was not found in Studora.")


def reset_document_processing_content(connection: psycopg.Connection, uploaded_document_id: str) -> None:
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


def batch_items(items: Sequence, size: int = INSERT_BATCH_SIZE):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def insert_document_sections(
    connection: psycopg.Connection,
    uploaded_document_id: str,
    sections: Sequence[ProcessedSection],
) -> None:
    if not sections:
        return

    now = datetime.now(UTC)
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO "UploadedDocumentSection"
                ("id", "uploadedDocumentId", "parentSectionId", "level", "title", "headingPath", "pageStart", "pageEnd", "sortOrder", "confidence", "metadata", "createdAt")
            VALUES
                (%s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT ("id") DO NOTHING
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


def update_document_section_page_end(
    connection: psycopg.Connection,
    section_id: str,
    page_end: int | None,
) -> None:
    if page_end is None:
        return

    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE "UploadedDocumentSection"
            SET "pageEnd" = %s
            WHERE "id" = %s::uuid
            """,
            (page_end, section_id),
        )


def insert_document_chunks(
    connection: psycopg.Connection,
    uploaded_document_id: str,
    chunks: Sequence[ProcessedChunk],
) -> None:
    if not chunks:
        return

    now = datetime.now(UTC)
    with connection.cursor() as cursor:
        for chunk_batch in batch_items(chunks):
            cursor.executemany(
                """
                INSERT INTO "UploadedDocumentChunk"
                    ("id", "uploadedDocumentId", "sectionId", "chunkIndex", "pageStart", "pageEnd", "text", "tokenCount", "contentPreview", "previousChunkId", "nextChunkId", "chunkHash", "heading", "headingPath", "sectionNumber", "metadata", "createdAt")
                VALUES
                    (%s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s::uuid, %s::uuid, %s, %s, %s, %s, %s::jsonb, %s)
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
                        chunk.chunk_hash,
                        chunk.heading,
                        chunk.heading_path,
                        chunk.section_number,
                        Jsonb(chunk.metadata),
                        now,
                    )
                    for chunk in chunk_batch
                ],
            )


def update_document_chunk_next_id(
    connection: psycopg.Connection,
    chunk_id: str,
    next_chunk_id: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE "UploadedDocumentChunk"
            SET "nextChunkId" = %s::uuid
            WHERE "id" = %s::uuid
            """,
            (next_chunk_id, chunk_id),
        )


def insert_document_summaries(
    connection: psycopg.Connection,
    uploaded_document_id: str,
    summaries: Sequence[ProcessedSummary],
) -> None:
    if not summaries:
        return

    now = datetime.now(UTC)
    with connection.cursor() as cursor:
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


def finish_document_processing(
    connection: psycopg.Connection,
    uploaded_document_id: str,
    *,
    chunk_count: int,
    outline: Sequence[dict[str, object]],
    page_count: int,
    section_count: int,
    status: str,
    summary_count: int,
    warnings: Sequence[str],
) -> None:
    now = datetime.now(UTC)
    with connection.cursor() as cursor:
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
                            "chunkCount": chunk_count,
                            "sectionCount": section_count,
                            "summaryCount": summary_count,
                            "processedBy": "studora-pdf-service",
                        }
                    }
                ),
                uploaded_document_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("Uploaded document could not be marked as ready.")


def save_document_chunks(
    connection: psycopg.Connection,
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

        for chunk_batch in batch_items(chunks):
            cursor.executemany(
                """
                INSERT INTO "UploadedDocumentChunk"
                    ("id", "uploadedDocumentId", "sectionId", "chunkIndex", "pageStart", "pageEnd", "text", "tokenCount", "contentPreview", "previousChunkId", "nextChunkId", "chunkHash", "heading", "headingPath", "sectionNumber", "metadata", "createdAt")
                VALUES
                    (%s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s::uuid, %s::uuid, %s, %s, %s, %s, %s::jsonb, %s)
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
                        chunk.chunk_hash,
                        chunk.heading,
                        chunk.heading_path,
                        chunk.section_number,
                        Jsonb(chunk.metadata),
                        now,
                    )
                    for chunk in chunk_batch
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


def mark_document_failed(connection: psycopg.Connection, uploaded_document_id: str, error: str) -> None:
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
