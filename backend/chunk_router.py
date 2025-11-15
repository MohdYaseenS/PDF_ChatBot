from fastapi import APIRouter
from pydantic import BaseModel
from chunk_and_vectorize.chunk_and_vectorize_utils import recursive_text_splitter, vectorize_text_chunks, create_faiss_index

chunk_router = APIRouter()

class PDF(BaseModel):
    text: str


@chunk_router.post("/chunk_and_vectorize")
async def chunk_and_vectorize(data: PDF):
    chunks = recursive_text_splitter(text = data.text, chunk_size = 1000, overlap = 200)
    chunk_vectors = vectorize_text_chunks(chunks = chunks)
    faiss_vector_index = create_faiss_index(vectors = chunk_vectors)







