"""
Educational RAG Assistant - Streamlit Application
Retrieval-Augmented Generation system powered by Groq API
Professional academic interface for document-based question answering
"""

import os
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import time
import numpy as np
from config import GROQ_API_KEY, MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, K_RETRIEVAL

# Set Groq API key from configuration
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


class SimpleEmbeddings(Embeddings):
    """
    Lightweight embeddings implementation using character frequency analysis.
    Avoids heavy ML dependencies (torch, transformers) while maintaining
    effective document retrieval capabilities.
    """
    
    def embed_documents(self, texts):
        """
        Embed a list of documents into vector representations.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self._text_to_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text):
        """
        Embed a single query text into a vector representation.
        
        Args:
            text: Query string to embed
            
        Returns:
            Embedding vector
        """
        return self._text_to_embedding(text)
    
    def _text_to_embedding(self, text):
        """
        Convert text to embedding using character frequency normalization.
        
        Args:
            text: Input text string
            
        Returns:
            Normalized embedding vector as list
        """
        text_lower = text.lower()
        embedding = np.zeros(256)
        
        # Count character frequencies
        for char in text_lower:
            if ord(char) < 256:
                embedding[ord(char)] += 1
        
        # Normalize vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()


# Configure Streamlit page settings
st.set_page_config(
    page_title="Educational RAG Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling - FIXED COLORS AND CONTRAST
st.markdown("""
    <style>
    /* Main background with professional dark theme */
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        min-height: 100vh;
    }
    
    /* Smooth fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Smooth slide animation */
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Header styling - FIXED: removed bottom border overlap */
    h1 {
        animation: fadeIn 0.6s ease-in;
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
        padding-bottom: 0px;
        border: none;
    }
    
    h2, h3 {
        animation: fadeIn 0.6s ease-in;
        color: #e8eef5;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    /* Subtitle text */
    p {
        color: #b0b8c1;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2d5a7b;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(45, 90, 123, 0.3);
    }
    
    .stButton > button:hover {
        background-color: #3a6fa3;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(45, 90, 123, 0.4);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #1e2635;
        border: 2px solid #3a4a5c;
        border-radius: 6px;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #7a8a9a;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #2d5a7b;
        box-shadow: 0 0 8px rgba(45, 90, 123, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, rgba(45, 90, 123, 0.1) 0%, rgba(52, 73, 94, 0.1) 100%);
    }
    
    /* Success message - FIXED: proper contrast */
    .stSuccess {
        animation: slideIn 0.5s ease-in;
        background-color: #1a3a2a;
        color: #7cfc7c;
        border: 1px solid #4a7a5a;
        border-radius: 6px;
        padding: 12px;
    }
    
    /* Warning message - FIXED: proper contrast */
    .stWarning {
        animation: slideIn 0.5s ease-in;
        background-color: #3a3a1a;
        color: #ffdd7c;
        border: 1px solid #7a7a3a;
        border-radius: 6px;
        padding: 12px;
    }
    
    /* Error message - FIXED: proper contrast */
    .stError {
        background-color: #3a1a1a;
        color: #ff7c7c;
        border: 1px solid #7a3a3a;
        border-radius: 6px;
        padding: 12px;
    }
    
    /* Info message - FIXED: proper contrast */
    .stInfo {
        background-color: #1a2a3a;
        color: #7cddff;
        border: 1px solid #3a5a7a;
        border-radius: 6px;
        padding: 12px;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #3a4a5c;
        margin: 25px 0;
    }
    
    /* Response box styling - FIXED: dark background with light text */
    .response-box {
        background: #1e2635;
        border-left: 4px solid #2d5a7b;
        padding: 20px;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        margin: 15px 0;
        animation: slideIn 0.4s ease-in;
        color: #e8eef5;
    }
    
    .response-box strong {
        color: #ffffff;
    }
    
    /* Chat history styling */
    .chat-message {
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
        background: #1e2635;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        border-left: 4px solid #34495e;
        color: #e8eef5;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #3a4a5c;
        border-radius: 6px;
        padding: 20px;
    }
    
    /* Label styling */
    label {
        color: #e8eef5;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar configuration panel
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")
    
    # Document management section
    st.markdown("### Document Management")
    
    # Load default documents button
    if st.button("Load Default Documents", use_container_width=True, key="load_defaults"):
        with st.spinner("Processing documents..."):
            try:
                docs_path = Path("data/documents")
                all_documents = []
                
                # Load all PDF files from documents directory
                for pdf_file in docs_path.glob("*.pdf"):
                    loader = PyPDFLoader(str(pdf_file))
                    documents = loader.load()
                    all_documents.extend(documents)
                    st.success(f"Loaded: {pdf_file.name}")
                
                if all_documents:
                    # Split documents into chunks for processing
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    chunks = text_splitter.split_documents(all_documents)
                    
                    # Initialize embeddings
                    st.session_state.embeddings = SimpleEmbeddings()
                    st.session_state.vector_store = FAISS.from_documents(
                        chunks,
                        st.session_state.embeddings
                    )
                    
                    # Initialize language model
                    st.session_state.llm = ChatGroq(
                        model=MODEL_NAME,
                        temperature=0.7,
                        groq_api_key=GROQ_API_KEY
                    )
                    
                    st.session_state.documents_loaded = True
                    st.success(f"Successfully processed {len(chunks)} document chunks")
                    st.info(f"Total documents loaded: {len(all_documents)}")
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")
    
    # Custom document upload section
    st.markdown("### Upload Custom Documents")
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files for analysis"
    )
    
    if uploaded_files and st.button("Process Uploaded Files", use_container_width=True, key="process_upload"):
        with st.spinner("Processing uploaded files..."):
            try:
                all_documents = []
                
                # Process each uploaded file
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    all_documents.extend(documents)
                    st.success(f"Loaded: {uploaded_file.name}")
                    
                    os.unlink(tmp_path)
                
                if all_documents:
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP
                    )
                    chunks = text_splitter.split_documents(all_documents)
                    
                    # Initialize embeddings
                    st.session_state.embeddings = SimpleEmbeddings()
                    st.session_state.vector_store = FAISS.from_documents(
                        chunks,
                        st.session_state.embeddings
                    )
                    
                    # Initialize language model
                    st.session_state.llm = ChatGroq(
                        model=MODEL_NAME,
                        temperature=0.7,
                        groq_api_key=GROQ_API_KEY
                    )
                    
                    st.session_state.documents_loaded = True
                    st.success(f"Processed {len(chunks)} document chunks")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # System information
    st.markdown("---")
    st.markdown("### System Information")
    st.info(f"Model: {MODEL_NAME}\nEmbeddings: Lightweight (No ML dependencies)")


# Main content area
st.markdown("""
    <div style="text-align: center; animation: fadeIn 0.8s ease-in; margin-bottom: 20px;">
        <h1>Educational RAG Assistant</h1>
        <p style="font-size: 16px; color: #b0b8c1; margin-top: 5px;">
            Retrieval-Augmented Generation for Document Analysis
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Check if documents are loaded
if not st.session_state.documents_loaded:
    st.warning("No documents loaded. Please load documents from the sidebar to begin.")
    st.markdown("""
        <div style="text-align: center; padding: 40px; background: #1e2635; border-radius: 8px; margin-top: 20px; border: 1px solid #3a4a5c;">
            <h3 style="color: #e8eef5;">Getting Started</h3>
            <p style="color: #b0b8c1;">Click "Load Default Documents" in the sidebar to load sample documents, or upload your own PDF files.</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.success("Documents ready for analysis")
    
    # Query interface section
    st.markdown("### Query Interface")
    
    # Display conversation history
    if st.session_state.chat_history:
        st.markdown("### Conversation History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Question {i+1}:** {question}")
            st.markdown(f"""
                <div class="response-box">
                    <strong>Answer:</strong><br>
                    {answer}
                </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
    
    # Question input area
    st.markdown("### Ask a Question")
    user_question = st.text_area(
        "Enter your question about the documents:",
        placeholder="Example: What are the key concepts discussed in the documents?",
        height=100,
        key="question_input"
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        if st.button("Search and Answer", use_container_width=True, key="search_btn"):
            if user_question.strip():
                with st.spinner("Processing query..."):
                    try:
                        if st.session_state.vector_store and st.session_state.llm:
                            start_time = time.time()
                            
                            # Create retriever from vector store
                            retriever = st.session_state.vector_store.as_retriever(
                                search_kwargs={"k": K_RETRIEVAL}
                            )
                            
                            # Define prompt template for RAG
                            prompt_template = PromptTemplate(
                                input_variables=["context", "question"],
                                template="""Based on the provided context, answer the following question accurately and comprehensively.

Context:
{context}

Question: {question}

Answer:"""
                            )
                            
                            # Function to format retrieved documents
                            def format_docs(docs):
                                return "\n\n".join(doc.page_content for doc in docs)
                            
                            # Build RAG chain
                            chain = (
                                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                                | prompt_template
                                | st.session_state.llm
                                | StrOutputParser()
                            )
                            
                            # Execute chain
                            response = chain.invoke(user_question)
                            elapsed_time = time.time() - start_time
                            
                            # Store in chat history
                            st.session_state.chat_history.append(
                                (user_question, response)
                            )
                            
                            # Display response
                            st.markdown("### Response")
                            st.markdown(f"""
                                <div class="response-box">
                                    {response}
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.success(f"Query processed in {elapsed_time:.2f} seconds")
                        else:
                            st.error("Model not loaded. Please load documents first.")
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("Please enter a question")
    
    with col2:
        if st.button("Clear History", use_container_width=True, key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col3:
        if st.button("Refresh", use_container_width=True, key="refresh_btn"):
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #7a8a9a; font-size: 12px;">
    <p>
        <strong>Educational RAG Assistant</strong><br>
        Advanced Retrieval-Augmented Generation System<br>
        Powered by Groq AI and LangChain
    </p>
</div>
""", unsafe_allow_html=True)