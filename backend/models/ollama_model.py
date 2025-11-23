"""
Ollama model implementation - lightweight local inference.
Ollama is easier to set up and faster than full transformers.
"""
from backend.core.config import ChatBotEnvConfig
from typing import Iterator, Optional
import logging
import requests
import json
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OllamaModel:
    """
    LLM wrapper using Ollama (local inference server).
    Ollama is lightweight and fast - perfect for local development.
    Install: https://ollama.ai
    """

    def __init__(self, config: ChatBotEnvConfig):
        self.config = config
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = config.model_id  # e.g., "llama2", "mistral", "phi"
        
        # Verify Ollama is running
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Ollama server connected at {self.base_url}")
        except Exception as e:
            logger.warning(f"Ollama server not reachable at {self.base_url}: {e}")
            logger.warning("Make sure Ollama is installed and running: https://ollama.ai")

    def generate(self, prompt: str) -> Optional[str]:
        """
        Non-streaming text generation (returns full response).
        """
        try:
            logger.info(f"Generating response via Ollama (non-stream): model={self.model_name}")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Streaming text generation (yields chunks as they arrive).
        """
        try:
            logger.info(f"Generating response via Ollama (streaming): model={self.model_name}")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                },
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Ollama streaming generation failed: {e}")
            yield f"\nError: Ollama generation failed - {e}"

