from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class ProcessRequest(BaseModel):
    uploaded_document_id: str = Field(alias="uploadedDocumentId")
    file_url: HttpUrl | None = Field(default=None, alias="fileUrl")
    file_path: str | None = Field(default=None, alias="filePath")
    source: str = "unknown"
    persist: bool = True


class ProcessedChunk(BaseModel):
    chunk_index: int = Field(alias="chunkIndex")
    page_start: int | None = Field(default=None, alias="pageStart")
    page_end: int | None = Field(default=None, alias="pageEnd")
    text: str
    token_count: int = Field(alias="tokenCount")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessResponse(BaseModel):
    uploaded_document_id: str = Field(alias="uploadedDocumentId")
    status: Literal["READY", "FAILED"]
    page_count: int = Field(alias="pageCount")
    chunk_count: int = Field(alias="chunkCount")
    chunks: list[ProcessedChunk]
    error: str | None = None

    model_config = {"populate_by_name": True}
