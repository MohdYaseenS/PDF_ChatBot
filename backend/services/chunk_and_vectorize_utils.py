# backend/services/chunk_and_vectorize.py
from typing import List, Tuple
import numpy as np
from pydantic import BaseModel, Field, field_validator
import logging
from backend.core.embeddings import embed_texts

logger = logging.getLogger("services.chunk_and_vectorize")

class TextChunkConfig(BaseModel):
    text: str = Field(..., min_length=1)
    chunk_size: int = Field(1000, gt=50)
    overlap: int = Field(200, ge=0, lt=1000)

    @field_validator("overlap")
    def validate_overlap(cls, overlap, values):
        if "chunk_size" in values and overlap >= values["chunk_size"]:
            raise ValueError("overlap must be smaller than chunk_size")
        return overlap

def recursive_text_splitter(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    cfg = TextChunkConfig(text=text, chunk_size=chunk_size, overlap=overlap)
    chunks = []
    start_idx = 0
    while start_idx < len(cfg.text):
        end_idx = start_idx + cfg.chunk_size
        chunk = cfg.text[start_idx:end_idx]
        chunks.append(chunk)
        # ensure forward progress
        start_idx = max(end_idx - cfg.overlap, end_idx)
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def chunk_and_vectorize(text: str, chunk_size: int = 1000, overlap: int = 200, model_name: str = "all-MiniLM-L6-v2") -> Tuple[List[str], np.ndarray]:
    chunks = recursive_text_splitter(text=text, chunk_size=chunk_size, overlap=overlap)
    vectors = embed_texts(chunks, model_name=model_name)
    return chunks, vectors
