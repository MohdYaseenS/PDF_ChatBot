import numpy as np
import faiss
import logging
from pydantic import BaseModel, Field, ValidationError, validator
from typing import List
from sentence_transformers import SentenceTransformer

# ---------------------------
# Logging (optional)
# ---------------------------
logger = logging.getLogger("chunk-vectorizer")
logger.setLevel(logging.INFO)



class TextChunkConfig(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to split.")
    chunk_size: int = Field(1000, gt=50, description="Max characters per chunk.")
    overlap: int = Field(200, ge=0, lt=1000, description="Overlap between chunks.")

    @validator("overlap")
    def validate_overlap(cls, overlap, values):
        if "chunk_size" in values and overlap >= values["chunk_size"]:
            raise ValueError("overlap must be smaller than chunk_size")
        return overlap

class VectorizeConfig(BaseModel):
    chunks: List[str] = Field(..., min_items=1, description="List of text chunks.")

    @validator("chunks")
    def validate_chunks(cls, chunks):
        if any(len(c.strip()) == 0 for c in chunks):
            raise ValueError("Chunks cannot contain empty strings.")
        return chunks

class FaissIndexConfig(BaseModel):
    vectors: np.ndarray

    @validator("vectors")
    def validate_vectors(cls, vectors):
        if not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be a NumPy array")

        if len(vectors.shape) != 2:
            raise ValueError("vectors must be a 2D NumPy array (N, D)")

        if vectors.shape[0] < 1:
            raise ValueError("vectors array must contain at least one vector")

        return vectors



# ======================================================
# Recursive text splitter
# ======================================================
def recursive_text_splitter(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap. Validated using Pydantic, and wrapped
    with safe exception handling.
    """
    try:
        cfg = TextChunkConfig(text=text, chunk_size=chunk_size, overlap=overlap)

        chunks = []
        start_idx = 0

        while start_idx < len(cfg.text):
            end_idx = start_idx + cfg.chunk_size
            chunk = cfg.text[start_idx:end_idx]
            chunks.append(chunk)
            start_idx = end_idx - cfg.overlap

        logger.info(f"Text successfully split into {len(chunks)} chunks.")
        return chunks

    except ValidationError as ve:
        logger.error(f"Validation error in recursive_text_splitter: {ve}")
        raise ValueError(f"Invalid input for text splitting: {ve}")

    except Exception as e:
        logger.error(f"Unexpected error splitting text: {e}")
        raise RuntimeError(f"Unexpected error splitting text: {e}")



# ======================================================
# Vectorization with Sentence Transformers
# ======================================================
def vectorize_text_chunks(chunks: List[str]) -> np.ndarray:
    """
    Convert text chunks to vectors. Validated and exception-safe.
    """
    try:
        cfg = VectorizeConfig(chunks=chunks)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = model.encode(cfg.chunks, show_progress_bar=False)

        logger.info(f"Vectorized {len(cfg.chunks)} chunks into shape {vectors.shape}.")
        return np.array(vectors)

    except ValidationError as ve:
        logger.error(f"Validation error in vectorize_text_chunks: {ve}")
        raise ValueError(f"Invalid chunk data: {ve}")

    except Exception as e:
        logger.error(f"Unexpected error vectorizing chunks: {e}")
        raise RuntimeError(f"Unexpected error vectorizing chunks: {e}")



# ======================================================
# FAISS index creation
# ======================================================
def create_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    """
    Create a FAISS index from validated vectors.
    """
    try:
        cfg = FaissIndexConfig(vectors=vectors)

        dimension = cfg.vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(cfg.vectors)

        logger.info(f"FAISS index created with {cfg.vectors.shape[0]} vectors.")
        return index

    except ValidationError as ve:
        logger.error(f"Validation error in create_faiss_index: {ve}")
        raise ValueError(f"Invalid vector data: {ve}")

    except Exception as e:
        logger.error(f"Unexpected error creating FAISS index: {e}")
        raise RuntimeError(f"Unexpected error creating FAISS index: {e}")