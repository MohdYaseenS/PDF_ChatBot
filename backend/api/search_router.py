# backend/api/search_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List
from backend.services.retrieval import FaissIndexWrapper, get_matches_from_indices
from backend.core.embeddings import embed_text

router = APIRouter()

# In-memory store (prototype)
_INDICES = {}  # key -> {"chunks": [...], "vectors": np.ndarray, "faiss": FaissIndexWrapper}

class BuildIndexRequest(BaseModel):
    key: str
    chunks: List[str]
    vectors: List[List[float]]

class QueryRequest(BaseModel):
    key: str
    query: str
    top_k: int = 3

@router.post("/build_index")
async def build_index(req: BuildIndexRequest):
    vectors = np.array(req.vectors, dtype=np.float32)
    fa = FaissIndexWrapper(vectors)
    _INDICES[req.key] = {"chunks": req.chunks, "vectors": vectors, "faiss": fa}
    return {"status": "ok", "n_chunks": len(req.chunks)}

@router.post("/query")
async def query_index(req: QueryRequest):
    if req.key not in _INDICES:
        raise HTTPException(status_code=404, detail="Index not found for key")
    store = _INDICES[req.key]
    qvec = embed_text(req.query)  # same model used for chunking
    D, I = store["faiss"].search(qvec, top_k=req.top_k)
    matches = get_matches_from_indices(store["chunks"], I)
    return {"matches": matches, "distances": D.tolist(), "indices": I.tolist()}
