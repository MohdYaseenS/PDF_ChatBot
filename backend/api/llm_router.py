# backend/api/llm_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.models.get_env_config import get_config

router = APIRouter()

class AnswerRequest(BaseModel):
    context: str
    question: str

@router.post("/answer")
async def generate_answer(req: AnswerRequest):
    cfg = get_config()
    if not req.context:
        raise HTTPException(status_code=400, detail="Context required")
    # TODO: Replace with real LLM call (OpenAI, local, etc.) using cfg.model_id etc.
    # Placeholder simple synthesis:
    answer = f"[placeholder answer] Question: {req.question}\nContext excerpt: {req.context[:500]}"
    return {"answer": answer}
