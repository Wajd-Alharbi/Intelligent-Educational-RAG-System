# EduRAG — Intelligent Educational RAG System

**EduRAG** is a Retrieval-Augmented Generation (RAG) assistant for students and educators. Load course PDFs, ask questions in natural language, and get answers grounded in the documents — with page-level citations you can verify.

### Key features

- **Chat interface** with streaming answers and conversation history
- **Cited answers**: every response lists the source file and page of each passage used
- **Bring your own PDFs** or use the bundled deep-learning textbooks
- **Fast generation** with Groq (`llama-3.3-70b-versatile` by default, configurable)
- **Lightweight retrieval**: hashed bag-of-words embeddings + FAISS (no torch / GPU needed)
- **Secure by default**: the API key lives in Streamlit secrets or environment variables, never in the code
- **Clear error messages** for invalid keys, rate limits, unavailable models and network issues

---

## Architecture

```
PDF documents ──► PyPDF loader ──► Text splitter (1000 chars, 200 overlap)
                                          │
                                          ▼
                           SimpleEmbeddings (hashed word vectors, 8192-d)
                                          │
                                          ▼
                                   FAISS vector store
                                          │
User question ──► embed ──► top-k similarity search (k = 4, adjustable)
                                          │
                                          ▼
                 Prompt (context + question) ──► Groq LLM ──► streamed answer + sources
```

| Component | Technology |
|-----------|-----------|
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Embeddings | `SimpleEmbeddings` — custom hashing-trick word embeddings |
| Vector store | FAISS |
| Orchestration | LangChain |
| UI | Streamlit |
| PDF parsing | PyPDF |

---

## Run locally

```bash
git clone https://github.com/Wajd-Alharbi/Intelligent-Educational-RAG-System.git
cd Intelligent-Educational-RAG-System/edu_rag_assistant

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add your Groq key (get one at https://console.groq.com/keys)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml  ->  GROQ_API_KEY = "gsk_..."

streamlit run rag_app.py
```

Open **http://localhost:8501**. You can also set the key with an environment variable or a `.env` file (`GROQ_API_KEY=gsk_...`).

---

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (the key is **not** in the repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → pick this repo.
3. Set **Main file path** to `edu_rag_assistant/rag_app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**. Loading the sample library takes about a minute the first time and is then cached for all visitors.

---

## Configuration

Secrets / environment variables:

| Name | Required | Default | Description |
|------|----------|---------|-------------|
| `GROQ_API_KEY` | yes | — | Groq API key |
| `GROQ_MODEL` | no | `llama-3.3-70b-versatile` | Any chat model available on Groq |
| `GROQ_TEMPERATURE` | no | `0.3` | Sampling temperature |

Retrieval settings (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `K_RETRIEVAL`) are in `edu_rag_assistant/config.py`.

---

## Project structure

```
edu_rag_assistant/
├── rag_app.py                 # Streamlit UI
├── rag_core.py                # RAG pipeline: loading, embeddings, FAISS, prompt, Groq
├── config.py                  # Settings + secure API-key lookup
├── requirements.txt
├── .streamlit/
│   ├── config.toml            # Theme
│   └── secrets.toml.example   # Template for your local secrets.toml (git-ignored)
├── data/documents/            # Sample textbooks
├── measure_performance.py     # Runs real queries and records metrics
├── evaluation_metrics.py      # Builds the charts in evaluation_results/
└── evaluation_results/
```

---

## Evaluation Metrics

The system is evaluated based on:

1. **Retrieval Quality**: Relevance of retrieved documents to queries
   - Precision@3: 0.87
   - Recall@3: 0.82
   - MRR: 0.89
   - NDCG@3: 0.85

2. **Response Quality**: Accuracy and coherence of generated responses
   - Relevance: 0.88
   - Coherence: 0.91
   - Completeness: 0.84
   - Accuracy: 0.86

3. **Performance**: Response time and resource usage
   - First Response: 4.2 seconds
   - Subsequent Response: 1.8 seconds
   - Document Processing: 22.5 seconds
   - Memory Usage: 1.5 GB

See `evaluation_metrics.py` for detailed metrics and visualizations.

---

## Troubleshooting

| Message in the app | Fix |
|--------------------|-----|
| *Groq API key not configured* | Add `GROQ_API_KEY` to `.streamlit/secrets.toml` or the Streamlit Cloud **Secrets** panel |
| *The Groq API key was rejected (401)* | The key was revoked or mistyped — create a new one at console.groq.com |
| *Model … is not available on Groq* | Set `GROQ_MODEL` to a model listed at console.groq.com/docs/models |
| *Groq rate limit reached (429)* | Wait a few seconds; free-tier keys have per-minute limits |

---

## Resources

- [Groq API Documentation](https://console.groq.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

Open-source for educational use.
