# backend/services/chunk_and_vectorize.py
from typing import List, Tuple
import numpy as np
from pydantic import BaseModel, Field, field_validator
import logging
from backend.core.embeddings import embed_texts
from backend.core.app_state import config  # centralized config
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("services.chunk_and_vectorize")

from pydantic import BaseModel, Field, field_validator

class TextChunkConfig(BaseModel):
    text: str = Field(..., min_length=1)
    chunk_size: int = Field(1000, gt=50)
    overlap: int = Field(200, ge=0, lt=1000)

    @field_validator("overlap")
    def validate_overlap(cls, overlap, info):
        data = info.data  # <- other validated fields
        chunk_size = data.get("chunk_size")

        if chunk_size is not None and overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        return overlap

def recursive_text_splitter(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    cfg = TextChunkConfig(text=text, chunk_size=chunk_size, overlap=overlap)
    chunks = []
    start_idx = 0
    text_len = len(cfg.text)
    while start_idx < text_len:
        end_idx = min(start_idx + cfg.chunk_size, text_len)
        chunk = cfg.text[start_idx:end_idx]
        chunks.append(chunk)
        start_idx = end_idx - cfg.overlap
        if start_idx < 0:
            start_idx = end_idx  # safety
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def lc_split(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(text)

def chunk_and_vectorize(text: str, chunk_size: int = None, overlap: int = None, model_name: str = None) -> Tuple[List[str], np.ndarray]:
    # default to config values
    chunk_size = chunk_size or config.chunk_size
    overlap = overlap or config.overlap
    model_name = model_name or config.embedding_model_id

    # chunks = recursive_text_splitter(text=text, chunk_size=chunk_size, overlap=overlap)
    chunks = lc_split(text=text, chunk_size=chunk_size, overlap=overlap)
    vectors = embed_texts(chunks, model_name=model_name)
    return chunks, vectors
