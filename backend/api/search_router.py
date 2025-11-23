# backend/api/search_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from backend.services.retrieval import FaissIndexWrapper, get_matches_from_indices
from backend.core.embeddings import embed_text

logger = logging.getLogger(__name__)
search_router = APIRouter()

# Thread pool for running synchronous embedding operations
embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")

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

@search_router.post("/build_index")
async def build_index(req: BuildIndexRequest):
    """Build FAISS index from chunks and vectors."""
    logger.info(f"Building index for key: {req.key} with {len(req.chunks)} chunks")
    try:
        vectors = np.array(req.vectors, dtype=np.float32)
        fa = FaissIndexWrapper(vectors)
        _INDICES[req.key] = {"chunks": req.chunks, "vectors": vectors, "faiss": fa}
        logger.info(f"Index built successfully for key: {req.key} (dim={vectors.shape[1]}, n_vectors={vectors.shape[0]})")
        return {"status": "ok", "n_chunks": len(req.chunks)}
    except Exception as e:
        logger.exception(f"Error building index for key {req.key}: {e}")
        raise HTTPException(status_code=500, detail=f"Error building index: {str(e)}")

@search_router.post("/query")
async def query_index(req: QueryRequest):
    """Query the FAISS index for similar chunks."""
    logger.info(f"Querying index for key: {req.key} (query: {req.query[:50]}..., top_k={req.top_k})")
    
    if req.key not in _INDICES:
        logger.warning(f"Index not found for key: {req.key}")
        raise HTTPException(status_code=404, detail="Index not found for key")
    
    store = _INDICES[req.key]
    
    try:
        # Run embedding generation in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        logger.debug("Generating query embedding (async)...")
        qvec = await loop.run_in_executor(embedding_executor, embed_text, req.query)
        logger.debug(f"Embedding generated (shape: {qvec.shape})")
        
        # FAISS search is fast and CPU-bound, but run in executor to be safe
        logger.debug("Performing FAISS search...")
        D, I = await loop.run_in_executor(
            None,  # Use default executor for CPU-bound operation
            store["faiss"].search,
            qvec,
            req.top_k
        )
        
        matches = get_matches_from_indices(store["chunks"], I)
        logger.info(f"Search complete: found {len(matches)} matches (distances: {D[0].tolist()})")
        
        return {"matches": matches, "distances": D.tolist(), "indices": I.tolist()}
    except Exception as e:
        logger.exception(f"Error querying index for key {req.key}: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying index: {str(e)}")
