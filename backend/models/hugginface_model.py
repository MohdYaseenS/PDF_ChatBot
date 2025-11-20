from backend.core.config import ChatBotEnvConfig
from typing import Iterator, Optional
import logging
from huggingface_hub import InferenceClient
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HugginFaceModel:
    """
    LLM wrapper using Hugginface InferenceClient.
    """

    def __init__(self, config: ChatBotEnvConfig):
        self.config = config
        self.client = InferenceClient(
            api_key=os.environ["HF_API_KEY"],
        )

    def generate(self, prompt: str) -> Optional[str]:
        """
        Non-streaming text generation (returns full response).
        """
        try:
            logger.info(f"Sending prompt to Together (non-stream): model={self.config.model_id}")
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Non-streaming request failed: {e}")
            return None

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Streaming text generation (yields chunks as they arrive).
        """
        try:
            logger.info(f"Sending prompt to Together (streaming): model={self.config.model_id}")
            stream = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True
            )

            for chunk in stream:
                if not chunk.choices:
                    continue  # skip if empty or missing choices
                delta = chunk.choices[0].delta.content or ""
                yield delta

        except Exception as e:
            logger.error(f"Streaming request failed: {e}")
            yield f"\nError: Streaming failed - {e}"