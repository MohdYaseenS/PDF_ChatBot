# backend/api/llm_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.core.response_generator import generate_response
from fastapi.responses import StreamingResponse
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

llm_router = APIRouter()

# Thread pool for running synchronous generators
executor = ThreadPoolExecutor(max_workers=2)

class AnswerRequest(BaseModel):
    context: str
    question: str

def build_optimized_prompt(context: str, question: str) -> str:
    """
    Builds an optimized prompt for better LLM responses.
    Uses structured format with clear instructions.
    """
    system_instruction = """You are a helpful AI assistant that answers questions based on the provided context. 
Your answers should be:
- Accurate and based only on the provided context
- Clear and well-structured
- Concise but complete
- If the context doesn't contain enough information, say so explicitly"""
    
    prompt = f"""{system_instruction}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    return prompt

def _consume_generator(generator):
    """Helper to consume synchronous generator and return list of chunks"""
    chunks = []
    try:
        for chunk in generator:
            if chunk:
                chunks.append(chunk)
    except Exception as e:
        logger.error(f"Error consuming generator: {e}")
        chunks.append(f"\n\nError: {str(e)}")
    return chunks

@llm_router.post("/answer")
async def generate_answer(req: AnswerRequest):
    # Use optimized prompt
    prompt = build_optimized_prompt(req.context, req.question)
    result = generate_response(prompt)

    if not result["success"]:
        return {"error": result["error"]}

    # Streaming flow
    if result.get("stream"):
        logger.info("Starting streaming response")
        
        async def streamer():
            try:
                # Get the synchronous generator
                generator = result["data"]
                
                # Iterate over the generator directly
                # This will block the event loop, but for streaming it's acceptable
                # as we want to yield chunks as soon as they arrive
                for chunk in generator:
                    if chunk:  # Only yield non-empty chunks
                        # FastAPI StreamingResponse expects bytes
                        chunk_bytes = chunk.encode('utf-8') if isinstance(chunk, str) else chunk
                        yield chunk_bytes
                        # Yield to event loop after each chunk to maintain responsiveness
                        await asyncio.sleep(0)
                    
            except Exception as e:
                logger.exception("Error in streamer")
                error_msg = f"\n\nError during streaming: {str(e)}"
                yield error_msg.encode('utf-8')
                
        return StreamingResponse(
            streamer(), 
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering in nginx
            }
        )

    # Normal flow
    logger.info("Returning non-streaming response")
    return {"answer": result["data"]}
