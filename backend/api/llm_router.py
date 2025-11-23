# backend/api/llm_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.core.response_generator import generate_response
from fastapi.responses import StreamingResponse
import asyncio
import logging

logger = logging.getLogger(__name__)

llm_router = APIRouter()


# ==============================
# Request Model
# ==============================
class AnswerRequest(BaseModel):
    context: str
    question: str


# ==============================
# Prompt Builder
# ==============================
def build_optimized_prompt(context: str, question: str) -> str:
    """
    Builds an optimized prompt for better LLM responses.
    Uses a structured and token-efficient format.
    """

    max_context_length = 5000  # character-based truncation
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"
        logger.warning("Context exceeded limit and was truncated.")

    system_instruction = "Answer based only on the provided context."

    prompt = f"""{system_instruction}

Context:
{context}

Q: {question}
A:"""

    return prompt


# ==============================
# Router Entry
# ==============================
@llm_router.post("/answer")
async def generate_answer(req: AnswerRequest):
    """
    Routes the question + context to the LLM.
    Supports both normal and streaming responses.
    """
    try:
        prompt = build_optimized_prompt(req.context, req.question)
        result = generate_response(prompt)
    except Exception as e:
        logger.exception("Error generating response from LLM backend.")
        raise HTTPException(status_code=500, detail=str(e))

    # -----------------------------
    # Non-streaming response
    # -----------------------------
    if not result.get("stream", False):
        logger.info("Returning non-streaming response.")
        return {"answer": result.get("data", "")}

    # -----------------------------
    # Streaming response
    # -----------------------------
    logger.info("Starting streaming LLM response.")

    async def streamer():
        """
        Wraps a synchronous generator and streams chunks asynchronously.
        """
        try:
            generator = result["data"]  # This is a Python generator

            for chunk in generator:
                if not chunk:
                    continue

                # FastAPI StreamingResponse requires bytes
                yield chunk.encode("utf-8")
                # Yield control to event loop after each chunk
                await asyncio.sleep(0)

        except Exception as e:
            logger.exception("Error during streaming LLM output.")
            yield f"\n\nError during streaming: {str(e)}".encode("utf-8")

    return StreamingResponse(
        streamer(),
        media_type="text/plain; charset=utf-8"
    )
