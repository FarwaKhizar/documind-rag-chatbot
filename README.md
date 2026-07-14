# 📄 DocuMind — RAG Chatbot

**Chat with your documents and get answers grounded in the source — with page-level citations.**

DocuMind turns a set of PDFs into an AI assistant that answers questions using *only* your documents, and shows exactly which page each answer came from. Built to demonstrate a clean, production-minded Retrieval-Augmented Generation (RAG) pipeline.

> Built by **Farwa Khizar** — AI / NLP Engineer specializing in RAG, semantic search, and LLM agents.

---

## ✨ What it does

- **Ask questions in natural language** over one or many PDFs
- **Grounded answers** — the model is constrained to the retrieved context and told to say when something isn't in the documents (this is what prevents hallucination)
- **Page-level citations** — every answer shows the source file, page number, and a match score
- **Upload your own PDFs** live, or use the included sample annual report

---

## 🧱 How it works

```
PDF ─► text extraction ─► chunking ─► embeddings ─► FAISS index
                                                        │
question ─► embed ─► similarity search (top-k) ─────────┘
                          │
                          ▼
        retrieved chunks + question ─► OpenAI ─► grounded answer + citations
```

| Stage | Tool | Why |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Runs locally, free, no API needed |
| Vector store | **FAISS** | Local & free; swaps cleanly to Pinecone / Qdrant / Weaviate in production |
| Answer generation | **OpenAI (GPT-4o-mini)** | Grounded via a strict context-only prompt |
| UI | **Streamlit** | Fast, clean, easy to run and deploy |

The retrieval half runs entirely offline — only the final answer step calls an external API.

---

## 🚀 Quick start

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd rag-chatbot-demo

# 2. Install
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env
#   then edit .env and paste your key from https://platform.openai.com/api-keys

# 4. Run
streamlit run app/streamlit_app.py
```

Then open the local URL Streamlit prints, click **Use sample annual report** (or upload your own PDF), and start asking questions.

**Try asking:**
- *How did revenue change year over year?*
- *What were the main risk factors?*
- *What is the outlook for next year?*

---

## 📁 Project structure

```
rag-chatbot-demo/
├── app/
│   ├── rag_pipeline.py     # RAG engine: ingest, embed, index, retrieve, answer
│   └── streamlit_app.py    # Styled Streamlit UI
├── data/
│   └── sample_annual_report.pdf
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Production notes

This is a focused demo, but the design mirrors production practice:

- **Swap the vector store** — FAISS → Pinecone / Qdrant / Weaviate by replacing the index class; the interface is small and isolated in `rag_pipeline.py`.
- **Scale** — the same pipeline pattern has been used on retrieval over 100M+ vectors.
- **Hybrid search & re-ranking** — semantic retrieval here can be extended with keyword (BM25) fusion and a re-ranking stage for higher precision.
- **Evaluation** — add an offline eval set (question → expected source) to track retrieval quality as documents change.

---

## 📬 Work with me

I build production RAG systems, semantic search, and AI agents. If you'd like something like this for your own documents or data, let's talk.
