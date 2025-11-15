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

# Recursive text splitter with overlap
def recursive_text_splitter(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap. Recursively splits if the chunk exceeds the chunk_size.
    """
    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = start_idx + chunk_size
        chunk = text[start_idx:end_idx]
        chunks.append(chunk)
        start_idx = end_idx - overlap  # Slide the window with overlap
    return chunks

# Vectorize the chunks using Sentence-Transformers
def vectorize_text_chunks(chunks: List[str]) -> np.ndarray:
    """
    Convert text chunks to vectors using Sentence-Transformers.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model
    vectors = model.encode(chunks, show_progress_bar=True)
    return np.array(vectors)

# Create FAISS index to store vectors
def create_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    """
    Create a FAISS index to store vectors and allow for similarity search.
    """
    dimension = vectors.shape[1]  # Dimensionality of the vectors
    index = faiss.IndexFlatL2(dimension)  # L2 distance-based index
    index.add(vectors)  # Add vectors to the index
    return index