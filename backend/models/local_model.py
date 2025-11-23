"""
Local LLM model implementation using transformers library.
Much faster than API calls - runs inference locally.
"""
from backend.core.config import ChatBotEnvConfig
from typing import Iterator, Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LocalModel:
    """
    Local LLM wrapper using transformers library.
    Runs inference on local GPU/CPU - much faster than API calls.
    """

    def __init__(self, config: ChatBotEnvConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading local model: {self.config.model_id} on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Create pipeline for easier generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    def generate(self, prompt: str) -> Optional[str]:
        """
        Non-streaming text generation (returns full response).
        """
        try:
            logger.info(f"Generating response locally (non-stream): model={self.config.model_id}")
            
            # Use pipeline for generation
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = outputs[0]["generated_text"].strip()
            return response
            
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            return None

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Streaming text generation (yields chunks as they arrive).
        Uses TextIteratorStreamer for proper streaming.
        """
        try:
            logger.info(f"Generating response locally (streaming): model={self.config.model_id}")
            
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Generate in separate thread
            generation_kwargs = {
                **inputs,
                "max_new_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer
            }
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Yield tokens as they arrive
            for token in streamer:
                yield token
                
        except Exception as e:
            logger.error(f"Local streaming generation failed: {e}")
            yield f"\nError: Local generation failed - {e}"

    def __del__(self):
        """Cleanup model from memory."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if self.pipeline is not None:
            del self.pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

