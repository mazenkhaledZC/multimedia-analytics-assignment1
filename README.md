# Multi-Modal RAG QA System — NASA Technical Reports

**DSAI 413: Multimedia Analytics — Assignment 1**  
**Student:** Mazen Khaled | **ID:** 202201534

A fully local, end-to-end **Multi-Modal Retrieval-Augmented Generation** system that answers questions over NASA Technical Reports by reasoning directly over **page images** — no OCR required.

---

## How It Works

```
NASA PDFs → Page Images → ColPali Embeddings → Qdrant (MaxSim)
                                                      ↓
User Query → ColPali Query Embedding → Top-K Pages → Qwen2-VL-2B → Answer
```

- **ColPali v1.2** embeds each page as patch-level multi-vectors (128-dim), capturing charts, figures, and tables visually.
- **Qdrant** stores multi-vectors and scores queries using late-interaction MaxSim — like ColBERT, but for images.
- **Qwen2-VL-2B-Instruct** (local, open-source, ~4GB) receives the retrieved page images and generates a cited answer.

---

## Features

- Visual retrieval over document images — no OCR, no text extraction
- Handles text, tables, charts, and figures natively
- Fully local — no external API keys required
- Streamlit web UI with configurable retrieval and generation settings
- Evaluation benchmark across 4 query modality types

---

## Project Structure

```
.
├── app.py                  # Streamlit frontend
├── evaluate.py             # Retrieval benchmark (text / table / figure / cross-modal)
├── download_dataset.py     # NASA NTRS dataset downloader
├── requirements.txt
├── docs/
│   └── report.md           # Full project report
├── src/
│   ├── ingest.py           # PDF → images → ColPali embeddings → Qdrant
│   ├── retrieval.py        # Query embedding + MaxSim search
│   └── generation.py       # Qwen2-VL-2B answer generation
├── showcase/
│   └── index.html          # Static showcase page
└── data/                   # Created at runtime (not tracked in git)
    ├── pdfs/
    ├── images/
    ├── qdrant_db/
    └── index/
```

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
# Add HF_TOKEN if needed for authenticated HuggingFace downloads
```

### 3. Download NASA reports

```bash
python download_dataset.py
```

### 4. Ingest PDFs into the vector database

```bash
python -m src.ingest
```

### 5. Launch the app

```bash
streamlit run app.py
```

### 6. Run the evaluation benchmark (optional)

```bash
python evaluate.py
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Retrieval model | [ColPali v1.2](https://huggingface.co/vidore/colpali-v1.2) |
| Vector database | [Qdrant](https://qdrant.tech/) (local, MultiVector MaxSim) |
| Generation model | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| PDF processing | pdf2image + Pillow |
| Frontend | Streamlit |
| Hardware | Apple Silicon (MPS), CUDA, or CPU |

---

## Evaluation

The benchmark covers 11 queries across four modality types:

| Type | Count |
|---|---|
| Text | 3 |
| Table | 3 |
| Figure / Chart | 3 |
| Cross-modal | 2 |

Results are saved to `data/eval_results.json` with per-query latency and top-1 MaxSim score.
