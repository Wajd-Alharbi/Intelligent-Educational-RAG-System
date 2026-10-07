"""
EduRAG - Educational RAG Assistant (Streamlit application)
Retrieval-Augmented Generation over PDF documents, powered by Groq.
"""

import time

import streamlit as st

from config import DOCUMENTS_PATH, K_RETRIEVAL, MODEL_NAME, get_groq_api_key
from rag_core import (
    build_vector_store,
    create_llm,
    describe_llm_error,
    load_directory,
    load_uploaded_files,
    retrieve,
    stream_answer,
)

st.set_page_config(
    page_title="EduRAG · Educational RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_QUESTIONS = [
    "What is backpropagation and why is it important?",
    "How does dropout help prevent overfitting?",
    "Explain the attention mechanism in transformers.",
    "What is the difference between SGD and Adam?",
]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stChatMessage { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 3.5rem; padding-bottom: 6rem; max-width: 1100px; }
    #MainMenu, footer { visibility: hidden; }

    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: radial-gradient(circle at top left, rgba(99,102,241,0.35), transparent 55%),
                    linear-gradient(135deg, #111827 0%, #0b1120 100%);
        border: 1px solid rgba(148,163,184,0.18);
        margin-bottom: 24px;
    }
    .hero h1 { font-size: 2.1rem; font-weight: 700; margin: 0 0 6px 0; color: #f8fafc; letter-spacing: -0.02em; }
    .hero p  { color: #94a3b8; margin: 0; font-size: 1.02rem; }
    .badge {
        display: inline-block; padding: 4px 12px; margin: 14px 8px 0 0; border-radius: 999px;
        font-size: 0.78rem; font-weight: 500; color: #c7d2fe;
        background: rgba(99,102,241,0.14); border: 1px solid rgba(99,102,241,0.35);
    }

    .feature-card {
        height: 100%; padding: 22px; border-radius: 14px;
        background: #111827; border: 1px solid rgba(148,163,184,0.16);
        transition: border-color .2s ease, transform .2s ease;
    }
    .feature-card:hover { border-color: rgba(99,102,241,0.6); transform: translateY(-2px); }
    .feature-card .icon { font-size: 1.6rem; }
    .feature-card h4 { margin: 10px 0 6px 0; color: #f1f5f9; font-size: 1.02rem; }
    .feature-card p  { margin: 0; color: #94a3b8; font-size: 0.92rem; line-height: 1.5; }

    .step { display: flex; gap: 14px; align-items: flex-start; margin: 12px 0; color: #cbd5e1; }
    .step .num {
        flex: 0 0 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        background: #6366f1; color: white; font-weight: 600; font-size: 0.85rem;
    }

    .status-pill { display: inline-flex; align-items: center; gap: 8px; font-size: 0.85rem; font-weight: 500; }
    .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .dot.ok { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
    .dot.off { background: #ef4444; box-shadow: 0 0 8px #ef4444; }

    .source-card {
        padding: 12px 14px; margin-bottom: 10px; border-radius: 10px;
        background: rgba(15,23,42,0.6); border-left: 3px solid #6366f1;
    }
    .source-card .meta { font-size: 0.8rem; font-weight: 600; color: #a5b4fc; margin-bottom: 4px; }
    .source-card .snippet { font-size: 0.85rem; color: #94a3b8; line-height: 1.45; }

    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    .app-footer { text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 40px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sample_library():
    """Index the bundled textbooks once per server process (shared across users)."""
    documents = load_directory(DOCUMENTS_PATH)
    if not documents:
        raise FileNotFoundError(f"No PDF files found in {DOCUMENTS_PATH}")
    store, chunk_count = build_vector_store(documents)
    sources = sorted({doc.metadata["source_name"] for doc in documents})
    return store, {"documents": len(sources), "pages": len(documents), "chunks": chunk_count, "names": sources}


@st.cache_resource(show_spinner=False)
def get_llm(api_key, model_name):
    return create_llm(api_key, model_name)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {"vector_store": None, "kb_stats": None, "kb_label": None, "messages": [], "pending_question": None}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

api_key = get_groq_api_key()


def escape_html(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_sources(sources):
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, src in enumerate(sources, start=1):
            page = f" · page {src['page']}" if src.get("page") else ""
            st.markdown(
                f"""<div class="source-card">
                        <div class="meta">[{i}] {escape_html(src['name'])}{page}</div>
                        <div class="snippet">{escape_html(src['snippet'])}</div>
                    </div>""",
                unsafe_allow_html=True,
            )


def set_knowledge_base(store, stats, label):
    st.session_state.vector_store = store
    st.session_state.kb_stats = stats
    st.session_state.kb_label = label
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 EduRAG")
    if api_key:
        st.markdown('<span class="status-pill"><span class="dot ok"></span>Groq API connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill"><span class="dot off"></span>Groq API key missing</span>', unsafe_allow_html=True)
    st.caption(f"Model: `{MODEL_NAME}`")

    st.divider()
    st.markdown("#### Knowledge base")
    source_choice = st.radio(
        "Source",
        ["Sample textbooks", "Upload my PDFs"],
        label_visibility="collapsed",
    )

    if source_choice == "Sample textbooks":
        st.caption("*Dive into Deep Learning* and *The Little Book of Deep Learning*.")
        if st.button("Load sample library", type="primary", use_container_width=True):
            with st.spinner("Indexing textbooks… the first load takes about a minute."):
                try:
                    store, stats = load_sample_library()
                    set_knowledge_base(store, stats, "Sample textbooks")
                    st.toast("Sample library is ready", icon="✅")
                except Exception as error:
                    st.error(f"Could not load the sample documents: {error}")
    else:
        uploaded_files = st.file_uploader(
            "PDF files", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
        )
        if st.button("Process files", type="primary", use_container_width=True, disabled=not uploaded_files):
            with st.spinner("Reading and indexing your documents…"):
                try:
                    documents = load_uploaded_files(uploaded_files)
                    store, chunk_count = build_vector_store(documents)
                    names = [f.name for f in uploaded_files]
                    stats = {"documents": len(names), "pages": len(documents), "chunks": chunk_count, "names": names}
                    set_knowledge_base(store, stats, "Uploaded documents")
                    st.toast("Your documents are ready", icon="✅")
                except Exception as error:
                    st.error(f"Could not process the files: {error}")

    if st.session_state.kb_stats:
        stats = st.session_state.kb_stats
        st.divider()
        st.markdown(f"#### Active: {st.session_state.kb_label}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Docs", stats["documents"])
        c2.metric("Pages", f"{stats['pages']:,}")
        c3.metric("Chunks", f"{stats['chunks']:,}")
        with st.expander("Indexed files"):
            for name in stats["names"]:
                st.markdown(f"- {name}")

    st.divider()
    st.markdown("#### Settings")
    top_k = st.slider("Passages per answer", min_value=2, max_value=8, value=K_RETRIEVAL)
    if st.button("🗑️ Clear conversation", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Educational RAG Assistant</h1>
        <p>Ask questions about your course material and get grounded answers with page-level citations.</p>
        <span class="badge">⚡ Groq LLM</span>
        <span class="badge">🔎 FAISS retrieval</span>
        <span class="badge">🦜 LangChain</span>
        <span class="badge">📄 PDF support</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key:
    st.error("**Groq API key not configured.** The assistant cannot answer questions until a key is added.", icon="🔑")
    with st.expander("How to add your key", expanded=True):
        st.markdown(
            """
1. Create a free key at [console.groq.com/keys](https://console.groq.com/keys).
2. **Running locally:** create `.streamlit/secrets.toml` next to `rag_app.py` containing
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
   (or put `GROQ_API_KEY=gsk_...` in a `.env` file).
3. **Streamlit Community Cloud:** open your app → **Settings → Secrets** and paste the same line.
4. Reload the page.
            """
        )

if st.session_state.vector_store is None:
    cols = st.columns(3)
    features = [
        ("📥", "Bring your documents", "Use the bundled deep-learning textbooks or upload your own lecture notes and papers."),
        ("🔎", "Grounded retrieval", "Relevant passages are located with vector search before the model writes a single word."),
        ("💬", "Cited answers", "Every answer links back to the pages it came from, so you can verify and study further."),
    ]
    for col, (icon, title, body) in zip(cols, features):
        col.markdown(
            f'<div class="feature-card"><div class="icon">{icon}</div><h4>{title}</h4><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Getting started")
    for i, text in enumerate(
        [
            "Choose a knowledge base in the sidebar: the sample textbooks or your own PDFs.",
            "Click <b>Load sample library</b> or <b>Process files</b> to index the documents.",
            "Ask a question in the chat box. Expand <b>Sources</b> under each answer to see the cited passages.",
        ],
        start=1,
    ):
        st.markdown(f'<div class="step"><span class="num">{i}</span><span>{text}</span></div>', unsafe_allow_html=True)
else:
    # Conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "🎓"):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])
            if message.get("elapsed"):
                st.caption(f"⏱ {message['elapsed']:.1f}s · {MODEL_NAME}")

    # Suggested questions for an empty chat on the sample library
    if not st.session_state.messages and st.session_state.kb_label == "Sample textbooks":
        st.markdown("##### Try asking")
        cols = st.columns(2)
        for i, question in enumerate(SAMPLE_QUESTIONS):
            if cols[i % 2].button(question, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

prompt = st.chat_input(
    "Ask a question about your documents…",
    disabled=st.session_state.vector_store is None or not api_key,
)
question = prompt or st.session_state.pop("pending_question", None)

if question and st.session_state.vector_store is not None and api_key:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🎓"):
        start = time.time()
        try:
            with st.spinner("Searching the documents…"):
                docs = retrieve(st.session_state.vector_store, question, k=top_k)
            response = st.write_stream(stream_answer(get_llm(api_key, MODEL_NAME), question, docs))
            elapsed = time.time() - start
            sources = [
                {
                    "name": doc.metadata.get("source_name", "document"),
                    "page": doc.metadata["page"] + 1 if isinstance(doc.metadata.get("page"), int) else None,
                    "snippet": " ".join(doc.page_content.split())[:320] + "…",
                }
                for doc in docs
            ]
            render_sources(sources)
            st.caption(f"⏱ {elapsed:.1f}s · {MODEL_NAME}")
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": sources, "elapsed": elapsed}
            )
        except Exception as error:
            message = describe_llm_error(error)
            st.error(message)
            st.session_state.messages.pop()  # drop the unanswered question

st.markdown(
    '<div class="app-footer">EduRAG · Retrieval-Augmented Generation for education · Built with Streamlit, LangChain, FAISS & Groq</div>',
    unsafe_allow_html=True,
)
