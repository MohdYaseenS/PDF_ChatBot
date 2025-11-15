from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from typing import List

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