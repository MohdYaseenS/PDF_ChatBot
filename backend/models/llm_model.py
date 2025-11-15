from pydantic import ValidationError
from backend.core.config import ChatBotEnvConfig

def get_llm_model():
    """
    Initializes and returns the appropriate LLM model based on environment variables.

    Returns:
        An instance of the selected LLM model.

    Raises:
        ValueError: If the backend is unsupported or environment validation fails.
    """
    try:
        config = ChatBotEnvConfig.from_env()
    except ValidationError as e:
        raise ValueError(f"Invalid LLM configuration: {e}")
    return config