# backend/core/embeddings.py
# Singleton wrapper to load SentenceTransformer ONCE and expose embedding helpers.

import threading
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("embeddings")

class _EmbeddingModel:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    @classmethod
    def get(cls, model_name: str = "all-MiniLM-L6-v2"):
        # Note: multiple calls with the same or different model_name will create only one instance.
        # If you need multiple embedding models simultaneously, adjust logic accordingly.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = _EmbeddingModel(model_name=model_name)
        return cls._instance

    def encode(self, texts, **kwargs) -> np.ndarray:
        vectors = self.model.encode(texts, show_progress_bar=False, **kwargs)
        return np.array(vectors, dtype=np.float32)


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> _EmbeddingModel:
    return _EmbeddingModel.get(model_name)


def embed_texts(texts, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    m = get_embedding_model(model_name)
    return m.encode(texts)


def embed_text(text, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    return embed_texts([text], model_name=model_name)[0]



def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> _EmbeddingModel:
    return _EmbeddingModel.get(model_name)


def embed_texts(texts, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    m = get_embedding_model(model_name)
    return m.encode(texts)


def embed_text(text, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    return embed_texts([text], model_name=model_name)[0]
