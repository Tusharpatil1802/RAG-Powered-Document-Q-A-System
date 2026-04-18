# 📄 DocMind — RAG-Powered Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system that lets you upload
documents (PDF, DOCX, TXT) and ask natural language questions — with answers grounded
in your content and full source citations.

---

## ✨ Features

- **Multi-format ingestion**: PDF, Word (DOCX), plain text
- **Smart chunking**: Recursive splitter with configurable size + overlap
- **Semantic search**: OpenAI `text-embedding-3-small` + ChromaDB vector store
- **Conversation memory**: Last 3 turns of chat included in every query
- **Source attribution**: Every answer shows which chunks were retrieved and similarity scores
- **Configurable**: Chunk size, overlap, top-k, and LLM model all adjustable in UI

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourname/docmind-rag.git
cd docmind-rag
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file (optional — you can also enter it in the UI):
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🗂 Project Structure

```
docmind-rag/
├── app.py              ← Streamlit UI + session management
├── rag_pipeline.py     ← Core RAG logic (ingest, embed, retrieve, generate)
├── utils.py            ← Token estimation, text cleaning, formatting
├── requirements.txt    ← All dependencies
├── .env                ← API keys (gitignored)
└── chroma_db/          ← Persistent vector store (auto-created)
```

---

## 🧠 How It Works (RAG Pipeline)

```
                    INGESTION (one-time)
                    ┌─────────────────────────────────────────┐
Upload file ──→ Load pages ──→ Split into chunks ──→ Embed chunks ──→ Store in ChromaDB
                    └─────────────────────────────────────────┘

                    QUERY (every question)
                    ┌─────────────────────────────────────────┐
User question ──→ Embed question ──→ Retrieve top-K chunks ──→ Build prompt ──→ GPT-4o ──→ Answer + sources
                    └─────────────────────────────────────────┘
```

### Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Splitter | `RecursiveCharacterTextSplitter` | Respects paragraph/sentence boundaries |
| Chunk size | 512 tokens (default) | Balance between context richness and retrieval precision |
| Chunk overlap | 64 tokens | Prevents answer fragments split across chunk boundaries |
| Embedding model | `text-embedding-3-small` | 5x cheaper than ada-002, comparable quality |
| Vector store | ChromaDB | Local, no infra, easy to swap to Pinecone for production |
| LLM temp | 0.1 | Factual, grounded — not creative |
| Context window | Last 3 turns | Conversation awareness without blowing token budget |

---

## 🎯 Use Cases

| Industry | Document type | Example query |
|---|---|---|
| Law firms | Contracts, NDAs | "What are the termination clauses?" |
| Startups | SOPs, handbooks | "What is the onboarding process for engineers?" |
| E-commerce | Product manuals | "What warranty does the X200 model carry?" |
| Research | Papers, reports | "What methodology did the study use?" |

---

## 📈 Upgrade Path (Production)

1. **Swap ChromaDB → Pinecone** for cloud-hosted, scalable vector search
2. **Add BM25 hybrid search** (keyword + semantic) for better domain-specific recall
3. **Add cross-encoder re-ranking** (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) for higher precision
4. **Add auth** with Streamlit-Authenticator or move to FastAPI + React
5. **Add RAGAS evaluation** to measure faithfulness and answer relevance automatically

---

## 💡 Tips for your Loom demo

1. Open with a relatable use case: "Imagine you have 50 contracts to review..."
2. Upload 2–3 PDFs live (contracts work great visually)
3. Ask a specific question that spans multiple docs
4. Show the source chunks panel — this is what clients pay for
5. Change a setting (top-k), re-query, show the difference

---

## 📄 License

MIT — use freely for portfolio, freelance, and commercial projects.
