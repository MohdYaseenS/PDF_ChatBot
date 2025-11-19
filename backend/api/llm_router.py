# backend/api/llm_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.core.response_generator import generate_response
import asyncio
from fastapi.responses import StreamingResponse

router = APIRouter()

class AnswerRequest(BaseModel):
    context: str
    question: str

@router.post("/answer")
async def generate_answer(req: AnswerRequest):
    prompt = f"You are provided with the following context: {req.context} Based on this context, answer the question: {req.question}"
    result = generate_response(prompt)

    if not result["success"]:
        return {"error": result["error"]}

    # Streaming flow
    if result.get("stream"):
        async def streamer():
            try:
                for chunk in result["data"]:  # generator from llm.generate_stream
                    yield chunk
                    await asyncio.sleep(0.01)
            except Exception as e:
                yield f"\n\nError during streaming: {str(e)}"
                
        return StreamingResponse(streamer(), media_type="text/plain")

    # Normal flow
    return {"answer": result["data"]}
