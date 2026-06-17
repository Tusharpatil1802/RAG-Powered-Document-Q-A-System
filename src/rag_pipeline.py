"""
rag_pipeline.py
===============
Core RAG logic: ingestion, embedding, retrieval, and generation.
Intentionally kept in one file for clarity and portfolio presentation.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from src.utils import estimate_tokens, clean_text


SYSTEM_PROMPT = """You are DocMind, an expert document analyst.
Answer questions strictly based on the provided context chunks.
Rules:
1. Only use information present in the context. Do not hallucinate.
2. If the answer is not in the context, say: "I couldn't find that in the uploaded documents."
3. Always cite the source document name when referencing information.
4. Be concise and precise. Use bullet points for lists.
5. For legal/contract documents, be especially careful and note any caveats."""

RAG_PROMPT_TEMPLATE = """Context from documents:
{context}

---
Chat History:
{chat_history}

---
User Question: {question}

Answer based on the context above:"""


class RAGPipeline:
    """
    End-to-end RAG pipeline with:
    - Multi-format document ingestion (PDF, DOCX, TXT)
    - Recursive text chunking with overlap
    - Gemini embeddings + ChromaDB vector store
    - Conversation-aware retrieval
    - Gemini generation with source attribution
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 4,
        model: str = "gemini-1.5-flash",
        persist_directory: str = "./chroma_db",
    ):
        google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Set it in your environment or enter it in the app."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.model = model
        self.persist_directory = persist_directory

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key,
        )

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,
            max_tokens=1024,
            google_api_key=google_api_key,
        )

        self.vectorstore: Optional[Chroma] = None
        self.doc_registry: Dict[str, Dict] = {}

    def ingest_files(self, uploaded_files) -> Dict[str, Any]:
        """
        Process uploaded Streamlit file objects → chunks → embeddings → ChromaDB.
        Returns stats dict for UI display.
        """
        all_docs: List[Document] = []

        for file in uploaded_files:
            docs = self._load_file(file)
            all_docs.extend(docs)

        if not all_docs:
            raise ValueError("No content could be extracted from uploaded files.")

        raw_chunks = self.splitter.split_documents(all_docs)
        chunks: List[Document] = []
        source_counters: Dict[str, int] = {}
        for chunk in raw_chunks:
            cleaned_content = clean_text(chunk.page_content)
            if not cleaned_content:
                continue

            src = chunk.metadata.get("source", "unknown")
            source_counters[src] = source_counters.get(src, 0) + 1
            chunk.page_content = cleaned_content
            chunk.metadata["chunk_id"] = source_counters[src]
            chunk.metadata["token_estimate"] = estimate_tokens(chunk.page_content)
            chunks.append(chunk)

        if not chunks:
            raise ValueError("No usable text remained after processing the uploaded files.")

        shutil.rmtree(self.persist_directory, ignore_errors=True)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="docmind_collection",
        )

        avg_tokens = (
            sum(c.metadata["token_estimate"] for c in chunks) // len(chunks)
            if chunks else 0
        )
        return {
            "num_docs": len(uploaded_files),
            "total_chunks": len(chunks),
            "avg_tokens": avg_tokens,
        }

    def _load_file(self, uploaded_file) -> List[Document]:
        """
        Route uploaded file to the correct LangChain loader.
        Streamlit gives us BytesIO objects, so we write to a temp file first.
        """
        suffix = Path(uploaded_file.name).suffix.lower()
        file_bytes = uploaded_file.getvalue()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyMuPDFLoader(tmp_path)
            elif suffix == ".docx":
                loader = Docx2txtLoader(tmp_path)
            elif suffix == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["file_type"] = suffix.lstrip(".")

            return docs

        finally:
            os.unlink(tmp_path)

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG query:
        1. Embed the question
        2. Retrieve top-k similar chunks
        3. Build prompt with context + history
        4. Generate answer
        5. Return answer + source metadata
        """
        if not self.vectorstore:
            raise RuntimeError("Knowledge base not built yet. Upload documents first.")

        results_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=self.top_k
        )
        if not results_with_scores:
            return {
                "answer": "I couldn't find that in the uploaded documents.",
                "sources": [],
                "tokens_used": estimate_tokens(question),
                "context_length": 0,
            }

        context_parts = []
        sources = []
        for i, (doc, score) in enumerate(results_with_scores):
            src = doc.metadata.get("source", "unknown")
            chunk_id = doc.metadata.get("chunk_id", i + 1)
            preview = doc.page_content[:200].replace("\n", " ").strip()

            context_parts.append(
                f"[Source: {src}, Chunk {chunk_id}]\n{doc.page_content}"
            )
            sources.append(
                {
                    "source": src,
                    "chunk_id": chunk_id,
                    "score": round(score, 3),
                    "preview": preview + ("…" if len(doc.page_content) > 200 else ""),
                }
            )

        context = "\n\n---\n\n".join(context_parts)

        history_str = ""
        if chat_history:
            history_lines = []
            for msg in chat_history[-6:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_lines.append(f"{role}: {msg['content'][:300]}")
            history_str = "\n".join(history_lines)

        user_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            chat_history=history_str or "No prior conversation.",
            question=question,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        answer = response.content

        tokens_used = estimate_tokens(user_prompt) + estimate_tokens(answer)

        return {
            "answer": answer,
            "sources": sources,
            "tokens_used": tokens_used,
            "context_length": len(context),
        }
