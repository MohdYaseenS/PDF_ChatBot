# backend/core/config.py
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class ChatBotEnvConfig(BaseModel):
    # LLM model (for generation)
    model_id: str = Field(..., min_length=1)
    # Embedding model (SentenceTransformers or HF sentence-transformers repo)
    embedding_model_id: str = Field(..., min_length=1)
    max_tokens: int = Field(512, gt=0)
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    stream_message: bool = Field(False)
    chunk_size: int = Field(1000, gt=0)
    overlap: int = Field(100, ge=0)
    # Backend url used by frontend / pdf processor if needed
    backend_url: str = Field("http://localhost:8081")

    @classmethod
    def from_env(cls):
        # pick up environment variables with fallbacks
        model_id = os.getenv("MODEL_ID", "moonshotai/Kimi-K2-Thinking")  # example default - adjust
        embedding_id = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return cls(
            model_id=model_id,
            embedding_model_id=embedding_id,
            max_tokens=int(os.getenv("MAX_TOKENS", 512)),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            stream_message=os.getenv("STREAM_MESSAGE", "true").lower() == "true",
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            overlap=int(os.getenv("OVERLAP", 100)),
            backend_url=os.getenv("BACKEND_URL", f"http://localhost:{os.getenv('PORT', '8081')}"),
        )

