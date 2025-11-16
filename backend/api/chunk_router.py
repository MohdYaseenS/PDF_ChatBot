# backend/api/chunk_router.py
from fastapi import APIRouter
from pydantic import BaseModel
from backend.models.get_env_config import get_config
from backend.services.chunk_and_vectorize import chunk_and_vectorize

router = APIRouter()

class PDFText(BaseModel):
    text: str

@router.post("/chunk")
async def chunk_endpoint(data: PDFText):
    cfg = get_config()
    chunks, vectors = chunk_and_vectorize(
        text=data.text,
        chunk_size=cfg.chunk_size,
        overlap=cfg.overlap,
        model_name=cfg.model_id,
    )
    return {
        "chunks": chunks,
        "vectors": vectors.tolist(),
        "n_vectors": vectors.shape[0],
        "vector_dim": vectors.shape[1],
    }
