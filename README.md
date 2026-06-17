# RAG-Powered Document Q&A System

A Streamlit app that lets you upload documents, build a local vector index, and ask grounded questions over the content using Gemini and ChromaDB.

## Overview

This project demonstrates a practical Retrieval-Augmented Generation workflow:

- upload `PDF`, `DOCX`, or `TXT` files
- split them into searchable chunks
- generate embeddings with Gemini
- store vectors locally in ChromaDB
- answer user questions with source-backed responses

It is designed as a portfolio-friendly project that is small enough to understand quickly while still showing the full RAG pipeline end to end.

## Features

- Multi-document ingestion for `PDF`, `DOCX`, and `TXT`
- Configurable chunk size, overlap, retrieval depth, and Gemini model
- Conversation-aware querying using the last few chat turns
- Source chunk previews with per-chunk relevance scores
- Local Chroma persistence for quick experimentation
- Streamlit interface for fast demos and iteration

## Tech Stack

- `Streamlit` for the UI
- `LangChain` for orchestration
- `Google Gemini` for embeddings and answer generation
- `ChromaDB` for local vector storage
- `PyMuPDF`, `docx2txt`, and `TextLoader` for document ingestion

## Project Structure

```text
RAG-Powered-Document-Q-A-System/
├── app.py
├── src/
│   ├── __init__.py
│   ├── rag_pipeline.py
│   └── utils.py
├── data/
│   └── sample_docs/
│       └── acme_employee_handbook.txt
├── docs/
│   └── RAG_Document_QA_Guide.pdf
├── requirements.txt
├── .env.example
└── .gitignore
```

## How It Works

```text
Documents -> Load text -> Split into chunks -> Create embeddings -> Store in ChromaDB
Question -> Retrieve top-k chunks -> Build prompt with context -> Gemini answer -> Show sources
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Tusharpatil1802/RAG-Powered-Document-Q-A-System.git
cd RAG-Powered-Document-Q-A-System
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add your Gemini API key:

```bash
cp .env.example .env
```

Then set:

```bash
GOOGLE_API_KEY=your_api_key_here
```

You can also paste the key directly into the Streamlit sidebar instead of using `.env`.

## Run Locally

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Sample Workflow

1. Launch the app.
2. Add your Gemini API key.
3. Upload one or more documents.
4. Click `Build Knowledge Base`.
5. Try the sample file in `data/sample_docs/acme_employee_handbook.txt`.
6. Ask questions such as:
   - `Summarize the main policies in this document.`
   - `What does the handbook say about employee leave?`
   - `Compare the onboarding steps across the uploaded files.`

## Current Limitations

- The app uses a simple chunking strategy without reranking.
- Retrieval is semantic only; there is no hybrid keyword search yet.
- The interface is optimized for demos and local experimentation, not multi-user deployment.

## Improvement Ideas

- Add hybrid search with BM25 + vector retrieval
- Add reranking for better answer precision
- Add document-level metadata filters
- Add evaluation with RAGAS or custom benchmarks
- Deploy with authentication and persistent cloud storage

## Why This Repo Is Useful

This project is a good example of:

- applied LLM engineering
- document intelligence workflows
- retrieval-augmented generation design
- rapid prototyping with modern GenAI tooling

## License

MIT
