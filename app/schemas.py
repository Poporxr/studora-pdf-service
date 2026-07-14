from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class ProcessRequest(BaseModel):
    uploaded_document_id: str = Field(alias="uploadedDocumentId")
    file_url: HttpUrl | None = Field(default=None, alias="fileUrl")
    file_path: str | None = Field(default=None, alias="filePath")
    source: str = "unknown"
    persist: bool = True


class ProcessedChunk(BaseModel):
    id: str | None = None
    chunk_index: int = Field(alias="chunkIndex")
    page_start: int | None = Field(default=None, alias="pageStart")
    page_end: int | None = Field(default=None, alias="pageEnd")
    section_id: str | None = Field(default=None, alias="sectionId")
    text: str
    token_count: int = Field(alias="tokenCount")
    content_preview: str | None = Field(default=None, alias="contentPreview")
    previous_chunk_id: str | None = Field(default=None, alias="previousChunkId")
    next_chunk_id: str | None = Field(default=None, alias="nextChunkId")
    chunk_hash: str | None = Field(default=None, alias="chunkHash")
    heading: str | None = None
    heading_path: list[str] = Field(default_factory=list, alias="headingPath")
    section_number: str | None = Field(default=None, alias="sectionNumber")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessedSection(BaseModel):
    id: str
    parent_section_id: str | None = Field(default=None, alias="parentSectionId")
    level: int
    title: str
    heading_path: list[str] = Field(alias="headingPath")
    page_start: int | None = Field(default=None, alias="pageStart")
    page_end: int | None = Field(default=None, alias="pageEnd")
    sort_order: int = Field(alias="sortOrder")
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessedSummary(BaseModel):
    id: str
    section_id: str | None = Field(default=None, alias="sectionId")
    kind: str
    title: str | None = None
    summary: str
    key_points: list[str] = Field(default_factory=list, alias="keyPoints")
    token_count: int | None = Field(default=None, alias="tokenCount")
    model: str | None = None
    source_version: str | None = Field(default=None, alias="sourceVersion")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=10)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimensions: int
    model: str


class ProcessResponse(BaseModel):
    uploaded_document_id: str = Field(alias="uploadedDocumentId")
    status: Literal[
        "READY",
        "READY_WITH_WARNINGS",
        "OCR_REQUIRED",
        "LOW_CONFIDENCE_EXTRACTION",
        "FAILED",
    ]
    page_count: int = Field(alias="pageCount")
    section_count: int = Field(default=0, alias="sectionCount")
    chunk_count: int = Field(alias="chunkCount")
    chunks: list[ProcessedChunk]
    sections: list[ProcessedSection] = Field(default_factory=list)
    summaries: list[ProcessedSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    model_config = {"populate_by_name": True}
