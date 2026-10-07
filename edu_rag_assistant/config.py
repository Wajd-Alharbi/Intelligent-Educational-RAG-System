"""
Configuration for the Educational RAG Assistant.

Secrets are NEVER stored in this file. The Groq API key is read, in order, from:
  1. Streamlit secrets  (.streamlit/secrets.toml locally, or the "Secrets" panel on Streamlit Cloud)
  2. Environment variables / a local .env file
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def _get_setting(name, default=None):
    """Read a setting from Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        # No secrets.toml present, or running outside Streamlit
        pass
    value = os.getenv(name)
    return value.strip() if value else default


def get_groq_api_key():
    """Return the Groq API key, or an empty string when it is not configured."""
    return _get_setting("GROQ_API_KEY", "") or ""


DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_model_name():
    """Return the Groq model to use. Read on every call so a change in secrets takes effect on the next rerun."""
    return _get_setting("GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


# Language Model Configuration
MODEL_NAME = get_model_name()
TEMPERATURE = float(_get_setting("GROQ_TEMPERATURE", "0.3"))

# Document Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVAL = 4

# Document Path (resolved relative to this file so the app works from any cwd)
DOCUMENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "documents")
