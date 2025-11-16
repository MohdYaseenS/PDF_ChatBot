# backend/services/retrieval.py
import faiss
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger("services.retrieval")

class FaissIndexWrapper:
    def __init__(self, vectors: np.ndarray):
        assert vectors.ndim == 2
        self.vectors = vectors.astype(np.float32)
        self.dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(self.vectors)
        logger.info(f"Built FAISS index with {self.vectors.shape[0]} vectors (dim={self.dim})")

    def search(self, qvec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(qvec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        D, I = self.index.search(q, top_k)
        return D, I

def get_matches_from_indices(chunks: List[str], indices: np.ndarray) -> List[str]:
    if indices.ndim == 2:
        indices = indices[0]
    return [chunks[i] for i in indices if i >= 0 and i < len(chunks)]
