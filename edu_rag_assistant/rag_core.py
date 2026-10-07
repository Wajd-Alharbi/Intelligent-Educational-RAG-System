"""
Core RAG pipeline: document loading, embeddings, vector store and answer generation.
Shared by the Streamlit app and the evaluation scripts.
"""

import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, K_RETRIEVAL, MODEL_NAME, TEMPERATURE

# pypdf logs a warning for every unusual font in the textbooks; keep the console clean
logging.getLogger("pypdf").setLevel(logging.ERROR)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "will",
    "with", "what", "which", "how", "why", "when", "who", "does", "do", "can",
}


class SimpleEmbeddings(Embeddings):
    """
    Lightweight lexical embeddings (no torch / transformers required).

    Each word is hashed into one of `dimensions` buckets ("hashing trick"),
    counts are dampened with log(1 + tf) and the vector is L2-normalised.
    Unlike character-frequency vectors, this captures the actual vocabulary
    of a chunk, so similarity search returns passages about the same concepts
    as the question.
    """

    def __init__(self, dimensions=8192):
        self.dimensions = dimensions

    def _bucket(self, token):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dimensions

    def _text_to_embedding(self, text):
        tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]
        buckets = np.fromiter((self._bucket(t) for t in tokens), dtype=np.int64, count=len(tokens))
        vector = np.log1p(np.bincount(buckets, minlength=self.dimensions).astype(np.float32))
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts):
        return [self._text_to_embedding(text) for text in texts]

    def embed_query(self, text):
        return self._text_to_embedding(text)


def load_pdf(path, display_name=None):
    """Load a PDF into LangChain documents, tagging each page with a readable source name."""
    documents = PyPDFLoader(str(path)).load()
    name = display_name or Path(path).name
    for doc in documents:
        doc.metadata["source_name"] = name
    return [doc for doc in documents if doc.page_content.strip()]


def load_directory(directory):
    """Load every PDF in a directory."""
    documents = []
    for pdf_file in sorted(Path(directory).glob("*.pdf")):
        documents.extend(load_pdf(pdf_file))
    return documents


def load_uploaded_files(uploaded_files):
    """Load Streamlit UploadedFile objects (written to temp files for PyPDF)."""
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        try:
            documents.extend(load_pdf(tmp_path, display_name=uploaded_file.name))
        finally:
            os.unlink(tmp_path)
    return documents


def build_vector_store(documents):
    """Split documents into chunks and index them in FAISS. Returns (store, chunk_count)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No extractable text was found in the provided PDF files.")
    store = FAISS.from_documents(chunks, SimpleEmbeddings())
    return store, len(chunks)


def create_llm(api_key, model_name=MODEL_NAME, temperature=TEMPERATURE):
    """Create the Groq chat model."""
    from langchain_groq import ChatGroq

    return ChatGroq(model=model_name, temperature=temperature, api_key=api_key, max_retries=2)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are EduRAG, a precise and friendly teaching assistant. Answer the student's "
            "question using ONLY the context excerpts below. If the context does not contain the "
            "answer, say so clearly instead of guessing. Structure the answer with short paragraphs "
            "or bullet points, and cite the excerpts you used as [1], [2], etc. Reply in the same "
            "language as the question.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_name", "document")
        page = doc.metadata.get("page")
        location = f"{source}, page {page + 1}" if isinstance(page, int) else source
        parts.append(f"[{i}] ({location})\n{doc.page_content}")
    return "\n\n".join(parts)


def retrieve(vector_store, question, k=K_RETRIEVAL):
    return vector_store.similarity_search(question, k=k)


def stream_answer(llm, question, docs):
    """Stream the answer token by token for the given retrieved documents."""
    chain = PROMPT | llm | StrOutputParser()
    return chain.stream({"context": format_context(docs), "question": question})


def answer(llm, question, docs):
    """Return the full answer as a string."""
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": format_context(docs), "question": question})


def describe_llm_error(error):
    """Turn Groq / network exceptions into a short actionable message."""
    text = str(error)
    lowered = text.lower()
    if "401" in text or "invalid api key" in lowered or "invalid_api_key" in lowered:
        return "The Groq API key was rejected (401). Create a new key at console.groq.com and update your secrets."
    if "429" in text or "rate limit" in lowered:
        return "Groq rate limit reached (429). Wait a moment and try again."
    if "model" in lowered and ("decommissioned" in lowered or "not found" in lowered or "does not exist" in lowered):
        return f"The model '{MODEL_NAME}' is not available on Groq. Set GROQ_MODEL in your secrets to a current model."
    if "connection" in lowered or "timeout" in lowered:
        return "Could not reach the Groq API. Check your internet connection and try again."
    return f"Unexpected error from the language model: {text}"
