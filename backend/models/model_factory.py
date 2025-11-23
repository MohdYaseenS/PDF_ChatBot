"""
Model factory to create the appropriate LLM model based on configuration.
Supports: HuggingFace API, Local Transformers, Ollama
"""
from backend.core.config import ChatBotEnvConfig
from backend.models.hugginface_model import HugginFaceModel
from backend.models.local_model import LocalModel
from backend.models.ollama_model import OllamaModel
from backend.models.together_model import TogetherModel

import logging
import os

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory to create LLM models based on configuration."""
    
    @staticmethod
    def create_model(config: ChatBotEnvConfig):
        """
        Create the appropriate model based on MODEL_TYPE environment variable.
        
        Options:
        - "api" or "hf_api": HuggingFace Inference API (default, slower, requires API key)
        - "local" or "transformers": Local model using transformers (faster, requires GPU/CPU)
        - "ollama": Ollama local server (fastest setup, requires Ollama installed)
        
        Returns:
            Model instance (HugginFaceModel, LocalModel, or OllamaModel)
        """
        model_type = os.getenv("MODEL_TYPE", "api").lower()
        
        if model_type in ["api", "hf_api", "huggingface_api"]:
            logger.info("Using HuggingFace Inference API")
            return HugginFaceModel(config)
        
        elif model_type in ["local", "transformers", "local_transformers"]:
            logger.info("Using local transformers model")
            try:
                return LocalModel(config)
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                logger.warning("Falling back to HuggingFace API")
                return HugginFaceModel(config)
        
        elif model_type in ['together']:
            logger.info("Using Together api model")
            try:
                return TogetherModel(config)
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                logger.warning("Falling back to HuggingFace API")
                return HugginFaceModel(config)

        
        elif model_type == "ollama":
            logger.info("Using Ollama local server")
            try:
                return OllamaModel(config)
            except Exception as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                logger.warning("Falling back to HuggingFace API")
                return HugginFaceModel(config)
        
        else:
            logger.warning(f"Unknown MODEL_TYPE: {model_type}, using HuggingFace API")
            return HugginFaceModel(config)

