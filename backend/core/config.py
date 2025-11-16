# backend/core/config.py
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class ChatBotEnvConfig(BaseModel):
    model_id: str = Field(..., min_length=1)
    max_tokens: int = Field(512, gt=0)
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    stream_message: bool = Field(False)
    chunk_size: int = Field(1000, gt=0)
    overlap: int = Field(100, ge=0)

    @classmethod
    def from_env(cls):
        model_id = os.getenv("MODEL_ID") or os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2"
        return cls(
            model_id=model_id,
            max_tokens=int(os.getenv("MAX_TOKENS", 512)),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            stream_message=os.getenv("STREAM_MESSAGE", "false").lower() == "true",
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            overlap=int(os.getenv("OVERLAP", 100)),
        )
