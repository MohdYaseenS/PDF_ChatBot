from backend.core.config import ChatBotEnvConfig
from typing import Iterator, Optional
import logging
import requests
import json
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HugginFaceModel:
    """
    LLM wrapper using HuggingFace Router API.
    Uses direct HTTP requests to router.huggingface.co
    """

    def __init__(self, config: ChatBotEnvConfig):
        self.config = config
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
        
        if not self.token:
            raise ValueError("HF_TOKEN or HF_API_KEY environment variable is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
        }

    def generate(self, prompt: str) -> Optional[str]:
        """
        Non-streaming text generation (returns full response).
        """
        try:
            logger.info(f"Sending prompt to HuggingFace Router (non-stream): model={self.config.model_id}")
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": self.config.model_id,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"Non-streaming request failed: {e}")
            return None

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Streaming text generation (yields chunks as they arrive).
        Uses Server-Sent Events (SSE) format.
        """
        try:
            logger.info(f"Sending prompt to HuggingFace Router (streaming): model={self.config.model_id}")
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": self.config.model_id,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": True,
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            # Parse SSE format
            for line in response.iter_lines():
                if not line:
                    continue
                
                # Decode line
                try:
                    line_str = line.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                
                # Skip non-data lines
                if not line_str.startswith("data:"):
                    continue
                
                # Check for end of stream
                if line_str.strip() == "data: [DONE]":
                    break
                
                try:
                    # Parse JSON from SSE data line
                    # Remove "data: " prefix and strip whitespace
                    json_str = line_str.lstrip("data:").strip()
                    if not json_str:
                        continue
                    
                    chunk_data = json.loads(json_str)
                    
                    # Extract content delta
                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                        delta = chunk_data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse SSE chunk: {line_str[:100]}, error: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    continue

        except Exception as e:
            logger.error(f"Streaming request failed: {e}")
            yield f"\nError: Streaming failed - {e}"