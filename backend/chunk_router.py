from fastapi import APIRouter
from pydantic import BaseModel
from chunk_and_vectorize.chunk_and_vectorize_utils import recursive_text_splitter, vectorize_text_chunks, create_faiss_index
from backend.models.llm_model import get_llm_model
from backend.core.config import ChatBotEnvConfig

llm_model=get_llm_model()

chunk_router = APIRouter()

class PDF(BaseModel):
    text: str

@chunk_router.post("/chunk_and_vectorize")
async def chunk_and_vectorize(data: PDF, llm_model: ChatBotEnvConfig):
    response={}
    chunks = recursive_text_splitter(text = data.text, chunk_size = llm_model.chunk_size, overlap = llm.overlap)
    chunk_vectors = vectorize_text_chunks(chunks = chunks)
    faiss_vector_index = create_faiss_index(vectors = chunk_vectors)

    response['chunks'] = chunks
    response['vectors'] = chunk_vectors
    response['faiss_index'] = faiss_vector_index

    return response


@chunk_router.get("/chunk_and_vectorize/health")
async def health_check():
    return {"status": "ok"}









