# backend/core/app_state.py
from backend.core.config import ChatBotEnvConfig
from backend.models.model_factory import ModelFactory
from backend.core.embeddings import _EmbeddingModel  # internal class, see below
import logging
import os

logger = logging.getLogger(__name__)

# Load configuration once
config = ChatBotEnvConfig.from_env()

# Create LLM client using factory (supports API, local, or Ollama)
llm = ModelFactory.create_model(config=config)

# Create embedding model singleton using embedding_model_id from config
# _EmbeddingModel is the class you already have in embeddings.py
embedding_model = _EmbeddingModel.get(model_name=config.embedding_model_id)

model_type = os.getenv("MODEL_TYPE", "api")
logger.info(f"App state initialized: model_type={model_type}, model={config.model_id}, embedding_model={config.embedding_model_id}")
