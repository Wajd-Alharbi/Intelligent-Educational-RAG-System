"""
Configuration file for Educational RAG Assistant
Groq API and model settings
"""

# Groq API Configuration
GROQ_API_KEY = ""

# Language Model Configuration
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.7

# Document Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVAL = 3

# Document Path
DOCUMENTS_PATH = "data/documents"
