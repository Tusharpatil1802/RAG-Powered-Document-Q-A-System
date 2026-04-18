"""
RAG-Powered Document Q&A System
================================
A production-ready Streamlit app for querying documents using RAG.
Supports PDF, DOCX, and TXT files.
"""

import streamlit as st
import os
import time
from pathlib import Path

from rag_pipeline import RAGPipeline
from utils import format_sources, estimate_tokens

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – RAG Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .source-card {
        background: #1e2130;
        border: 1px solid #2d3149;
        border-left: 3px solid #4f8ef7;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
    }
    .metric-chip {
        display: inline-block;
        background: #1a2744;
        color: #4f8ef7;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        margin: 2px;
    }
    h1 { color: #e8eaf6 !important; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ──────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get yours at aistudio.google.com"
    )

    st.divider()
    st.markdown("### 📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="PDF, TXT, and DOCX supported"
    )

    st.divider()
    st.markdown("### 🔧 RAG Settings")

    chunk_size = st.slider("Chunk size (tokens)", 256, 1024, 512, 64,
                           help="Larger = more context per chunk, slower retrieval")
    chunk_overlap = st.slider("Chunk overlap", 0, 200, 64, 16,
                              help="Overlap between consecutive chunks")
    top_k = st.slider("Top-K chunks retrieved", 1, 10, 4,
                      help="How many chunks to retrieve per query")
    model = st.selectbox("LLM", ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.5-pro"],
                         help="Gemini flash models are extremely fast and ideal for RAG context.")

    st.divider()

    if st.button("🚀 Build Knowledge Base", type="primary", use_container_width=True):
        if not api_key:
            st.error("Enter your Gemini API key first.")
        elif not uploaded_files:
            st.error("Upload at least one document.")
        else:
            os.environ["GOOGLE_API_KEY"] = api_key
            with st.spinner("Ingesting documents…"):
                try:
                    rag = RAGPipeline(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k=top_k,
                        model=model,
                    )
                    stats = rag.ingest_files(uploaded_files)
                    st.session_state.rag = rag
                    st.session_state.docs_loaded = True
                    st.session_state.doc_stats = stats
                    st.session_state.chat_history = []
                    st.success(f"✅ Indexed {stats['total_chunks']} chunks from {stats['num_docs']} docs")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if st.session_state.docs_loaded:
        stats = st.session_state.doc_stats
        st.markdown(f"""
        **Knowledge Base Stats**
        - 📄 Documents: `{stats.get('num_docs', 0)}`
        - 🧩 Chunks: `{stats.get('total_chunks', 0)}`
        - 🔤 Avg chunk tokens: `{stats.get('avg_tokens', 0)}`
        """)

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── Main area ───────────────────────────────────────────────────────────────────
st.markdown("# 📄 DocMind — RAG Document Q&A")
st.markdown("Ask questions about your uploaded documents. Answers are grounded in your content.")

if not st.session_state.docs_loaded:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1** — Enter your Gemini API key in the sidebar")
    with col2:
        st.info("**Step 2** — Upload PDFs, DOCX, or TXT files")
    with col3:
        st.info("**Step 3** — Click 'Build Knowledge Base' and start asking")
    st.stop()

# ── Chat interface ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📚 {len(msg['sources'])} source chunks used"):
                for src in msg["sources"]:
                    st.markdown(f"""
<div class="source-card">
<strong>📄 {src['source']}</strong> &nbsp;
<span class="metric-chip">chunk #{src['chunk_id']}</span>
<span class="metric-chip">score: {src['score']:.2f}</span>
<br/><br/>
<em>{src['preview']}</em>
</div>
""", unsafe_allow_html=True)

# ── Query input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about your documents…"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            start = time.time()
            try:
                result = st.session_state.rag.query(
                    prompt,
                    chat_history=st.session_state.chat_history[:-1]
                )
                elapsed = time.time() - start

                st.markdown(result["answer"])
                st.caption(f"⚡ {elapsed:.2f}s · {result['tokens_used']} tokens · {len(result['sources'])} chunks retrieved")

                if result["sources"]:
                    with st.expander(f"📚 {len(result['sources'])} source chunks used"):
                        for src in result["sources"]:
                            st.markdown(f"""
<div class="source-card">
<strong>📄 {src['source']}</strong> &nbsp;
<span class="metric-chip">chunk #{src['chunk_id']}</span>
<span class="metric-chip">score: {src['score']:.2f}</span>
<br/><br/>
<em>{src['preview']}</em>
</div>
""", unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })

            except Exception as e:
                error_msg = f"Query failed: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                })

if __name__ == '__main__':
    import os, sys
    print("Redirecting to: streamlit run app.py")
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", sys.argv[0]] + sys.argv[1:])
