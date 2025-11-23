from together import Together
from backend.core.config import LLMEnvConfig
from typing import Iterator, Optional
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TogetherModel:
    """
    LLM wrapper using Together's official Python SDK.
    """

    def __init__(self, config: LLMEnvConfig):
        self.config = config
        self.client = Together(api_key=self.config.api_key)

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