"""
Multi-Modal RAG QA System — NASA Technical Reports
Powered by ColPali (visual late-interaction retrieval) + moondream2 (local OSS VLM)
"""

import os
import sys
import torch
import streamlit as st
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="NASA Docs QA — ColPali RAG",
    page_icon="🚀",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚀 NASA Docs RAG")
    st.caption("ColPali · Qdrant · moondream2")
    st.divider()
    top_k = st.slider("Pages to retrieve", 1, 8, 4)
    max_gen_pages = st.slider("Pages to send to LLM", 1, 3, 2)
    st.divider()
    st.markdown(
        "**Dataset:** NASA Technical Reports  \n"
        "**Retrieval:** ColPali v1.2 (MaxSim)  \n"
        "**Vector DB:** Qdrant (local)  \n"
        "**Generator:** Qwen2-VL-2B-Instruct (local, open-source)"
    )


# ── Cached model loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ColPali retriever...")
def load_retriever():
    from src.retrieval import build_retriever
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return build_retriever(qdrant_path="data/qdrant_db", device=device)


@st.cache_resource(show_spinner="Loading Qwen2-VL-2B (first run downloads ~4GB)...")
def load_generator():
    from src.generation import build_generator
    return build_generator()


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_index_exists() -> bool:
    return Path("data/qdrant_db/collection/nasa_colpali").exists()


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("Multi-Modal Document QA")
st.markdown("Ask questions about NASA Technical Reports — answers grounded in retrieved page images.")

if not check_index_exists():
    st.warning("No index found. Run `python download_dataset.py` then `python -m src.ingest`.")
    st.stop()

try:
    retriever = load_retriever()
    stats = retriever.get_collection_stats()
    st.success(f"Index ready — {stats['total_pages']} pages across NASA technical reports.")
except Exception as e:
    st.error(f"Failed to load retriever: {e}")
    st.stop()

_, answer_fn = load_generator()

st.divider()
query = st.text_input(
    "Ask a question about the documents",
    placeholder="e.g. What thrust levels are achieved in the propulsion tests?",
)
search_btn = st.button("Search & Answer", type="primary", disabled=not query)

if search_btn and query:
    with st.spinner("Retrieving relevant pages with ColPali..."):
        results = retriever.retrieve(query, top_k=top_k)

    if not results:
        st.warning("No relevant pages found.")
        st.stop()

    with st.spinner("Generating answer with Qwen2-VL-2B..."):
        try:
            output = answer_fn(query, results, max_pages=max_gen_pages)
            answer_text = output["answer"]
        except Exception as e:
            answer_text = f"Generation failed: {e}"

    st.subheader("Answer")
    st.markdown(answer_text)

    st.divider()
    st.subheader(f"Retrieved Pages (top {len(results)})")
    cols = st.columns(min(len(results), 4))
    for col, page in zip(cols, results[:4]):
        with col:
            img_path = page.get("image_path", "")
            if img_path and Path(img_path).exists():
                st.image(Image.open(img_path), use_container_width=True)
            else:
                st.markdown("_(image not available)_")
            st.caption(
                f"**{page['doc_name']}**  \n"
                f"Page {page['page_num']} / {page['total_pages']}  \n"
                f"Score: {page['score']:.4f}"
            )

    if len(results) > 4:
        with st.expander(f"Show remaining {len(results) - 4} results"):
            for page in results[4:]:
                st.markdown(
                    f"- **{page['doc_name']}** · Page {page['page_num']} · Score: {page['score']:.4f}"
                )
