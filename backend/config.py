import os
from dotenv import load_dotenv

# Load env variables from .env file if it exists
load_dotenv()

# Hugging Face token configuration
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# Default model settings
DEFAULT_MODEL = "google/gemma-4-E2B-it"

# Server configuration
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "127.0.0.1")

# LangSmith Observability configurations
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "").strip()
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "chatbot-observability")

def get_hf_token(override_token: str = "") -> str:
    """
    Returns the Hugging Face token, prioritizing user override (from UI)
    and falling back to the environment variable.
    """
    token = override_token.strip() if override_token else ""
    if not token:
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
    return token
