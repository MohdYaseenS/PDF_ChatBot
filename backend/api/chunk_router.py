from fastapi import APIRouter
from pydantic import BaseModel
from chunk_and_vectorize.chunk_and_vectorize_utils import recursive_text_splitter, vectorize_text_chunks, create_faiss_index
from backend.models.get_env_config import get_config


chunk_router = APIRouter()

class PDF(BaseModel):
    text: str


@chunk_router.post("/chunk_and_vectorize")
async def chunk_and_vectorize(data: PDF):
    response={}
    chatbot_env_config = get_config()
    chunks = recursive_text_splitter(text = data.text, chunk_size = chatbot_env_config.chunk_size, overlap = chatbot_env_config.overlap)
    chunk_vectors = vectorize_text_chunks(chunks = chunks)
    faiss_vector_index = create_faiss_index(vectors = chunk_vectors)

    response['chunks'] = chunks
    response['vectors'] = chunk_vectors
    response['faiss_index'] = faiss_vector_index

    return response
    







