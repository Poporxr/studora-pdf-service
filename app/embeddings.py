import threading

from fastembed import TextEmbedding

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384

_model: TextEmbedding | None = None
_model_lock = threading.Lock()


def _ensure_model_loaded() -> TextEmbedding:
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            _model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        return _model


def preload_embedding_model() -> None:
    _ensure_model_loaded()


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _ensure_model_loaded()
    return [embedding.tolist() for embedding in model.embed(texts)]
