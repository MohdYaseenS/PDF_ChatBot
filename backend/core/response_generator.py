from backend.models.hugginface_model import HugginFaceModel
from backend.models.get_env_config import get_config
import logging

config = get_config()
llm = HugginFaceModel(config=config)
logger = logging.getLogger(__name__)

def generate_response(prompt: str) -> str:
    """
    Generates a non-streaming response from the LLM.

    Args:
        prompt (str): The input prompt for the LLM.
    Returns:
        str: The generated response from the LLM.
    """

    try:
        if llm.config.stream_message:
            # Return a generator for streaming
            return {
                "success": True,
                "data": llm.generate_stream(prompt),
                "stream": True,
                "error": None
            }
        else:
            # Normal one-shot generation
            output = llm.generate(prompt)
            if output:
                return {
                    "success": True,
                    "data": output,
                    "stream": False,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "data": None,
                    "stream": False,
                    "error": "No response from LLM."
                }
    except Exception as e:
        logger.exception("Failed to generate explanation.")
        return {
            "success": False,
            "data": None,
            "stream": llm.config.stream_message,
            "error": str(e)
        }



