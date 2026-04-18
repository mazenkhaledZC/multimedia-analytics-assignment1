# Assignment 1 — Multi-Modal RAG QA System
## DSAI 413: Multimedia Analytics

**Student:** Mazen Khaled  
**ID:** 202201534  
**Date:** April 18, 2026

---

## 1. Overview

This project implements a **Multi-Modal Retrieval-Augmented Generation (RAG)** question-answering system over a corpus of NASA Technical Reports. Unlike traditional text-only RAG pipelines, this system operates directly on **page images**, enabling retrieval that captures visual content such as charts, diagrams, figures, and tables — without any OCR step.

The pipeline is composed of three stages:

```
NASA PDFs → Page Images → ColPali Embeddings → Qdrant (vector DB)
                                                      ↓
User Query → ColPali Query Embedding → MaxSim Search → Top-K Pages
                                                      ↓
                              Qwen2-VL-2B-Instruct (VLM) → Answer
```

---

## 2. System Architecture

### 2.1 Data Ingestion (`src/ingest.py`)

PDF documents are downloaded from NASA's Technical Reports Server (NTRS) and converted to page images at 150 DPI using `pdf2image`. Each page image is then embedded using **ColPali v1.2**, which produces multi-vector (patch-level) representations of size 128 dimensions per patch.

These patch embedding lists are stored in a **Qdrant** local vector database configured with `MultiVectorComparator.MAX_SIM` — enabling ColBERT-style late-interaction scoring at retrieval time. Each vector point carries a payload with document name, PDF path, page number, total pages, and image file path.

Key design choices:
- **No OCR required** — ColPali directly encodes visual content.
- **Multi-vector storage** — each page is represented as a list of patch vectors, not a single embedding.
- **MaxSim scoring** — the query-document score is computed as the sum of maximum cosine similarities between query token embeddings and document patch embeddings.

### 2.2 Dataset Download (`download_dataset.py`)

The downloader queries the NTRS search API across five technical domains:

| Domain | Example Topics |
|---|---|
| Propulsion | System performance, thrust efficiency |
| Aerodynamics | Computational fluid dynamics |
| Thermal Systems | Spacecraft thermal control |
| Robotics | Autonomous Mars rover systems |
| Structures | Aircraft structural fatigue analysis |

Up to 4 documents per query are downloaded, deduplicated, and saved to `data/pdfs/`.

### 2.3 Retrieval (`src/retrieval.py`)

At query time, the user's text query is embedded with ColPali's query encoder, producing a list of token-level vectors. Qdrant's `query_points` endpoint performs MaxSim scoring across all stored page embeddings and returns the top-K most relevant pages along with their metadata.

The `ColPaliRetriever` class exposes:
- `retrieve(query, top_k)` — single query retrieval
- `retrieve_multi_query(queries, top_k)` — batched multi-query retrieval
- `get_collection_stats()` — collection diagnostics

### 2.4 Generation (`src/generation.py`)

The retrieved top pages are passed — as actual images — to **Qwen2-VL-2B-Instruct**, a 2-billion parameter open-source vision-language model (~4 GB in bfloat16). For each page, the model receives:

1. The page image
2. A prompt specifying the document name, page number, and the user's question

Answers from multiple pages are concatenated with per-page citations (e.g., `[nasa_19920020863, p.334]`). Generation parameters:

| Parameter | Value |
|---|---|
| `max_new_tokens` | 300 |
| `temperature` | 0.7 |
| `top_p` | 0.9 |
| `repetition_penalty` | 1.5 |
| `no_repeat_ngram_size` | 4 |

No external API key is required — the VLM runs fully locally on MPS (Apple Silicon), CUDA, or CPU.

### 2.5 Application (`app.py`)

The frontend is a **Streamlit** web application with:
- A sidebar for configuring `top_k` (pages to retrieve, 1–8) and `max_gen_pages` (pages sent to VLM, 1–3).
- A text input field for the user's question.
- Display of the generated answer with formatted citations.
- A visual grid showing the retrieved page images with their document name, page number, and relevance score.
- Cached model loading via `@st.cache_resource` to avoid reloading on each interaction.

---

## 3. Evaluation (`evaluate.py`)

A benchmark suite of 11 queries was designed to cover four distinct modality types:

| Type | Count | Example Query |
|---|---|---|
| Text | 3 | "What are the main objectives of this research?" |
| Table | 3 | "What are the performance values in the results table?" |
| Figure | 3 | "What does the pressure vs velocity graph show?" |
| Cross-modal | 2 | "Compare the experimental results shown in the figure with the data in the table." |

The evaluator measures per-query **retrieval latency**, **top-1 score**, and the **retrieved page list** (top 3), then saves the full results to `data/eval_results.json`. Summary metrics reported: total queries, average latency (seconds), average top-1 MaxSim score, and query count per type.

---

## 4. Technology Stack

| Component | Technology |
|---|---|
| Retrieval model | ColPali v1.2 (`vidore/colpali-v1.2`) |
| Vector database | Qdrant (local, on-disk) |
| Generation model | Qwen2-VL-2B-Instruct (local, open-source) |
| PDF processing | pdf2image + Pillow |
| Frontend | Streamlit |
| Hardware target | Apple Silicon (MPS), CUDA, or CPU |

---

## 5. Project Structure

```
Assignment_1/
├── app.py                  # Streamlit frontend
├── evaluate.py             # Retrieval benchmark suite
├── download_dataset.py     # NTRS dataset downloader
├── requirements.txt        # Python dependencies
├── docs/
│   └── report.md           # This report
├── src/
│   ├── ingest.py           # PDF → embeddings → Qdrant
│   ├── retrieval.py        # ColPali query + MaxSim search
│   └── generation.py       # Qwen2-VL-2B answer generation
├── data/
│   ├── pdfs/               # Downloaded NASA PDFs
│   ├── images/             # Extracted page images (JPEG)
│   ├── qdrant_db/          # Qdrant local vector store
│   └── index/              # Per-document JSON metadata
└── showcase/
    └── index.html          # Static showcase page
```

---

## 6. Key Design Decisions

**Why ColPali instead of CLIP or text embeddings?**  
ColPali produces patch-level multi-vectors per page rather than a single pooled embedding. This preserves local spatial structure and enables late-interaction MaxSim scoring — meaning a query about "the bar chart on the right side" can match the specific image patches containing that chart, rather than relying on a coarse page-level similarity.

**Why a local VLM (Qwen2-VL-2B) instead of a hosted API?**  
Using a fully local open-source model removes dependency on external API keys or network access, keeps the system self-contained, and avoids cost per query. At 2B parameters it is lightweight enough to run on a MacBook with Apple Silicon via the MPS backend.

**Why Qdrant with MultiVector support?**  
Qdrant is one of the few vector databases with native multi-vector (ColBERT-style) collection support, making it a natural fit for ColPali's output format without needing custom aggregation logic outside the database.

---

## 7. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NASA reports
python download_dataset.py

# 3. Ingest PDFs into the vector database
python -m src.ingest

# 4. Launch the Streamlit app
streamlit run app.py

# 5. (Optional) Run the evaluation benchmark
python evaluate.py
```

A HuggingFace token (`HF_TOKEN`) can be set in `.env` for authenticated model downloads.

---

## 8. Conclusion

This assignment demonstrates an end-to-end multi-modal RAG system that goes beyond simple text retrieval by treating document pages as visual objects. By combining ColPali's patch-level visual embeddings with Qdrant's MaxSim multi-vector search and a local vision-language model for generation, the system can answer questions grounded in figures, charts, and tables — content that would be lost or degraded in text-only pipelines.
