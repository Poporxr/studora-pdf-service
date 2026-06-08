from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    internal_job_secret: str
    database_url: str | None = None
    max_pdf_mb: int = 30
    max_pdf_pages: int = 250
    temporary_max_pdf_mb: int = 20
    temporary_max_pdf_pages: int = 120
    extraction_batch_pages: int = 20
    max_chunk_tokens: int = 800
    chunk_overlap_tokens: int = 120
    request_timeout_seconds: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
