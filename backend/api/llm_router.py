# backend/api/llm_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.models.get_env_config import get_config
import os
from huggingface_hub import InferenceClient

router = APIRouter()

class AnswerRequest(BaseModel):
    context: str
    question: str

@router.post("/answer")
async def generate_answer(req: AnswerRequest):
    cfg = get_config()
    if not req.context:
        raise HTTPException(status_code=400, detail="Context required")


    client = InferenceClient(
        api_key=os.environ["HF_API_KEY"],
    )

    completion = client.chat.completions.create(
        model=cfg.model_id,
        messages=[
            {
                "role": "user",
                "content": f"Answer the question:{req.question} based on the context:\nContext: {req.context}"
            }
        ],
    )
    answer=completion.choices[0].message
    return {"answer": answer}
