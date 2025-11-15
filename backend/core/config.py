import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class ChatBotEnvConfig(BaseModel):
    model_id: str = Field(..., min_length=1, description="Model ID for the selected backend")
    max_tokens: int = Field(512, gt=0, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature")
    stream_message: bool = Field(False, description="Enable streaming responses")
    chunk_size: int = Field(1000, description="Size of each chunk")
    overlap: int = Field(100, description= "Overlap size for each consecutive chunks")

    @classmethod
    def from_env(cls):
        return cls(
            backend=os.getenv("LLM_BACKEND"),
            api_key=os.getenv("API_KEY"),
            model_id=os.getenv("MODEL_ID"),
            max_tokens=int(os.getenv("MAX_TOKENS", 512)),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            stream_message=os.getenv("STREAM_MESSAGE", "false").lower() == "true",
        )