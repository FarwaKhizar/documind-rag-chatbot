"""
streamlit_app.py
----------------
A polished RAG chatbot UI: ask questions about a document set and get
answers grounded in the sources, with page-level citations.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys

# Silence noisy (harmless) TensorFlow / CUDA registration logs before imports.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from rag_pipeline import RAGEngine  # noqa: E402

load_dotenv()

# ------------------------------- Page setup -------------------------------- #

st.set_page_config(
    page_title="DocuMind — RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------- Brand CSS --------------------------------- #

NAVY = "#1F3864"
MINT = "#9FE1CB"
MINT_DEEP = "#0F6E56"

st.markdown(
    f"""
    <style>
    .stApp {{ background: #F7F8FA; }}
    /* Header band */
    .brand-header {{
        background: {NAVY};
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 18px;
        display: flex; align-items: center; justify-content: space-between;
    }}
    .brand-left {{ display:flex; align-items:center; gap:12px; }}
    .brand-badge {{
        width:40px; height:40px; border-radius:10px; background:{MINT};
        display:flex; align-items:center; justify-content:center; font-size:22px;
    }}
    .brand-name {{ color:#fff; font-size:20px; font-weight:600; margin:0; }}
    .brand-sub {{ color:{MINT}; font-size:13px; margin:0; }}
    .brand-tag {{
        background:#E1F5EE; color:{MINT_DEEP}; font-size:13px; font-weight:600;
        padding:6px 14px; border-radius:999px;
    }}
    /* Citation chip */
    .cite {{
        display:inline-block; background:#E1F5EE; color:{MINT_DEEP};
        font-size:12px; font-weight:500; padding:4px 10px; border-radius:8px;
        margin:4px 6px 0 0;
    }}
    .src-box {{
        background:#fff; border:1px solid #E4E7EC; border-left:4px solid {MINT};
        border-radius:8px; padding:10px 14px; margin:6px 0; font-size:13px;
        color:#475467;
    }}
    .answer-box {{
        background:#fff; border:1px solid #E4E7EC; border-radius:12px;
        padding:18px 20px; font-size:15px; color:#1D2939; line-height:1.6;
    }}
    div[data-testid="stMetricValue"] {{ color:{NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Header ------------------------------------ #

st.markdown(
    f"""
    <div class="brand-header">
      <div class="brand-left">
        <div class="brand-badge">📄</div>
        <div>
          <p class="brand-name">DocuMind — Chat With Your Documents</p>
          <p class="brand-sub">Farwa Khizar · RAG demo · answers grounded in your sources</p>
        </div>
      </div>
      <span class="brand-tag">● Powered by RAG + OpenAI</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Session state --------------------------------- #

if "engine" not in st.session_state:
    st.session_state.engine = None
if "history" not in st.session_state:
    st.session_state.history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


@st.cache_resource(show_spinner=False)
def get_engine() -> RAGEngine:
    """Load the engine once (embedding model load is the slow part)."""
    return RAGEngine(api_key=os.getenv("OPENAI_API_KEY"))


def index_files(paths: list[str]) -> None:
    engine = get_engine()
    # rebuild fresh so re-uploading doesn't double-count
    engine._chunks = []
    for p in paths:
        engine.add_pdf(p)
    engine.build_index()
    st.session_state.engine = engine
    st.session_state.indexed_files = engine.sources


# ------------------------------ Sidebar ------------------------------------ #

with st.sidebar:
    st.subheader("📁 Documents")

    key_ok = bool(os.getenv("OPENAI_API_KEY"))
    if key_ok:
        st.success("OpenAI API key detected")
    else:
        st.warning("No OPENAI_API_KEY found — retrieval works, but answers are disabled. Add it to your .env.")

    uploaded = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    use_sample = st.button("Use sample annual report", use_container_width=True)

    st.divider()
    st.caption("Upload a PDF or use the sample, then ask questions in the chat.")

# Handle sample
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
if use_sample:
    sample = os.path.join(DATA_DIR, "sample_annual_report.pdf")
    if not os.path.exists(sample):
        sample = os.path.join(DATA_DIR, "test_report.pdf")
    if os.path.exists(sample):
        with st.spinner("Indexing sample report…"):
            index_files([sample])
        st.toast("Sample report indexed")
    else:
        st.error("No sample PDF found in /data. Drop one in or upload above.")

# Handle uploads
if uploaded:
    os.makedirs("/tmp/rag_uploads", exist_ok=True)
    paths = []
    for uf in uploaded:
        dest = os.path.join("/tmp/rag_uploads", uf.name)
        with open(dest, "wb") as f:
            f.write(uf.getbuffer())
        paths.append(dest)
    with st.spinner(f"Indexing {len(paths)} document(s)…"):
        index_files(paths)
    st.toast(f"Indexed {len(paths)} document(s)")

# --------------------------- Loaded indicator ------------------------------ #

engine = st.session_state.engine
if st.session_state.indexed_files:
    st.caption("📎 Ready — loaded: " + ", ".join(st.session_state.indexed_files))
    st.divider()

# ------------------------------ Chat area ---------------------------------- #

if engine is None:
    st.info(
        "👈 Upload a PDF or click **Use sample annual report** to begin. "
        "Then ask questions like *“What were the main risk factors?”* or "
        "*“How did revenue change year over year?”*"
    )
else:
    # show history
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            st.markdown(f'<div class="answer-box">{turn["a"]}</div>', unsafe_allow_html=True)
            if turn["sources"]:
                st.markdown("**Sources**", help="The exact chunks this answer was grounded in.")
                for s in turn["sources"]:
                    st.markdown(
                        f'<div class="src-box"><span class="cite">📄 {s["source"]} · p.{s["page"]} '
                        f'· {s["score"]:.0%} match</span><br>{s["snippet"]}</div>',
                        unsafe_allow_html=True,
                    )

    question = st.chat_input("Ask a question about your documents…")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and reasoning…"):
                ans = engine.ask(question, k=4)
            st.markdown(f'<div class="answer-box">{ans.text}</div>', unsafe_allow_html=True)
            if ans.sources:
                st.markdown("**Sources**")
                srcs = []
                for r in ans.sources:
                    snippet = r.chunk.text[:180].strip() + "…"
                    st.markdown(
                        f'<div class="src-box"><span class="cite">📄 {r.chunk.source} · p.{r.chunk.page} '
                        f'· {r.score:.0%} match</span><br>{snippet}</div>',
                        unsafe_allow_html=True,
                    )
                    srcs.append(
                        {"source": r.chunk.source, "page": r.chunk.page,
                         "score": r.score, "snippet": snippet}
                    )
        st.session_state.history.append({"q": question, "a": ans.text, "sources": srcs})
