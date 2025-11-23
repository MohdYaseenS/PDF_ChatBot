# backend/api/chunk_router.py
from fastapi import APIRouter
from pydantic import BaseModel
from backend.core.app_state import config
from backend.services.chunk_and_vectorize import chunk_and_vectorize

chunk_router = APIRouter()

class PDFText(BaseModel):
    text: str

@chunk_router.post("/chunk")
async def chunk_endpoint(data: PDFText):
    cfg = config
    chunks, vectors = chunk_and_vectorize(
        text=data.text,
        chunk_size=cfg.chunk_size,
        overlap=cfg.overlap,
        model_name=cfg.embedding_model_id,  # Fixed: use embedding_model_id, not model_id
    )
    return {
        "chunks": chunks,
        "vectors": vectors.tolist(),
        "n_vectors": vectors.shape[0],
        "vector_dim": vectors.shape[1],
    }
