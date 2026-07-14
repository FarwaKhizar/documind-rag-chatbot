"""
rag_pipeline.py
---------------
Core Retrieval-Augmented Generation (RAG) pipeline.

Flow:
    PDF -> text chunks -> embeddings -> FAISS index
    question -> retrieve top-k chunks -> Gemini answers using ONLY those chunks
    -> answer + source citations (file name + page number)

Design notes:
- Embeddings run locally with sentence-transformers (free, no API key).
- Vector store is FAISS (local, free). Swaps cleanly to Pinecone/Qdrant in prod.
- Only the LLM answer step calls an external API (OpenAI).
- The model is instructed to answer strictly from retrieved context and to say
  when the answer isn't in the documents. This is what keeps RAG grounded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ----------------------------- Data structures ----------------------------- #

@dataclass
class Chunk:
    """A single retrievable piece of a document."""
    text: str
    source: str          # file name
    page: int            # 1-indexed page number


@dataclass
class RetrievedChunk:
    """A chunk returned by search, with its similarity score."""
    chunk: Chunk
    score: float


@dataclass
class Answer:
    """The final answer plus the chunks it was grounded in."""
    text: str
    sources: list[RetrievedChunk]


# ------------------------------- The engine -------------------------------- #

class RAGEngine:
    """
    Builds a searchable index from PDFs and answers questions against it.

    Typical use:
        engine = RAGEngine(api_key=...)
        engine.add_pdf("data/annual_report.pdf")
        engine.build_index()
        answer = engine.ask("What were the main risk factors?")
    """

    def __init__(
        self,
        api_key: str | None = None,
        embed_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "gpt-4o-mini",
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> None:
        self.embed_model_name = embed_model_name
        self.llm_model_name = llm_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Local embedding model — loaded once and reused.
        self._embedder = SentenceTransformer(embed_model_name)

        # Configure OpenAI if a key is available.
        key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=key) if key else None

        self._chunks: list[Chunk] = []
        self._index: faiss.IndexFlatIP | None = None

    # ---- Ingestion ---- #

    def add_pdf(self, path: str) -> int:
        """Extract, chunk, and stage a PDF for indexing. Returns #chunks added."""
        reader = PdfReader(path)
        source = os.path.basename(path)
        added = 0
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            for piece in self._split(text):
                self._chunks.append(Chunk(text=piece, source=source, page=page_num))
                added += 1
        return added

    def _split(self, text: str) -> list[str]:
        """Simple character-based splitter with overlap to preserve context."""
        words = text.split()
        chunks, current, length = [], [], 0
        for word in words:
            current.append(word)
            length += len(word) + 1
            if length >= self.chunk_size:
                chunks.append(" ".join(current))
                # keep an overlap tail so context isn't cut mid-thought
                overlap_words, ol = [], 0
                for w in reversed(current):
                    ol += len(w) + 1
                    overlap_words.insert(0, w)
                    if ol >= self.chunk_overlap:
                        break
                current, length = overlap_words, ol
        if current:
            chunks.append(" ".join(current))
        return chunks

    # ---- Indexing ---- #

    def build_index(self) -> None:
        """Embed all staged chunks and build the FAISS similarity index."""
        if not self._chunks:
            raise ValueError("No documents added. Call add_pdf() first.")
        vectors = self._embed([c.text for c in self._chunks])
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)   # inner product on normalized vecs = cosine
        index.add(vectors)
        self._index = index

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.astype("float32")

    # ---- Retrieval + generation ---- #

    def retrieve(self, question: str, k: int = 4) -> list[RetrievedChunk]:
        if self._index is None:
            raise ValueError("Index not built. Call build_index() first.")
        q_vec = self._embed([question])
        scores, idxs = self._index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=float(score)))
        return results

    def ask(self, question: str, k: int = 4) -> Answer:
        """Retrieve context and have Gemini answer strictly from it."""
        retrieved = self.retrieve(question, k=k)
        context = "\n\n".join(
            f"[Source: {r.chunk.source}, page {r.chunk.page}]\n{r.chunk.text}"
            for r in retrieved
        )
        prompt = (
            "You are a precise assistant answering questions about the provided "
            "documents. Use ONLY the context below. If the answer is not in the "
            "context, say you couldn't find it in the documents. Be concise and "
            "factual. Do not invent numbers.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )
        if self._client is None:
            return Answer(
                text="(No API key set. Add OPENAI_API_KEY to your .env to enable answers.)",
                sources=retrieved,
            )

        # Retry with exponential backoff — handles transient 429 rate limits.
        import time

        last_err = None
        for attempt in range(4):
            try:
                response = self._client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "system", "content":
                            "You are a precise assistant that answers strictly from the "
                            "provided context and never invents facts."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                return Answer(
                    text=response.choices[0].message.content.strip(),
                    sources=retrieved,
                )
            except Exception as exc:
                last_err = exc
                if "429" in str(exc) or "rate limit" in str(exc).lower():
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s
                    continue
                break  # non-rate-limit error: don't retry
        return Answer(text=f"(Model error: {last_err})", sources=retrieved)

    # ---- Convenience ---- #

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self._chunks})
