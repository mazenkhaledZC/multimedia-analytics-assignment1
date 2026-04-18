"""
Ingestion pipeline: PDF → page images → ColPali embeddings → Qdrant.
ColPali treats each page as an image and produces multi-vector (patch-level)
embeddings, enabling visual late-interaction retrieval without OCR.
"""

import os
import json
import torch
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    MultiVectorConfig,
    MultiVectorComparator,
    PointStruct,
    PayloadSchemaType,
)

# ColPali outputs 128-dim patch embeddings
COLPALI_DIM = 128
COLLECTION_NAME = "nasa_colpali"
IMAGE_DIR = "data/images"
INDEX_DIR = "data/index"


def load_colpali(device: str = None) -> tuple[ColPali, ColPaliProcessor]:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ColPali on {device}...")
    hf_token = os.environ.get("HF_TOKEN")
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.bfloat16,
        device_map=device,
        token=hf_token,
    ).eval()
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2", token=hf_token)
    return model, processor


def pdf_to_images(pdf_path: str, dpi: int = 150) -> List[Image.Image]:
    """Convert PDF pages to PIL images."""
    images = convert_from_path(pdf_path, dpi=dpi, fmt="RGB")
    return images


def embed_images(
    images: List[Image.Image],
    model: ColPali,
    processor: ColPaliProcessor,
    batch_size: int = 2,
    device: str = "cpu",
) -> List[List[List[float]]]:
    """
    Embed page images with ColPali.
    Returns: list of pages, each page is a list of patch embeddings (float lists).
    """
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor.process_images(batch).to(device)
        with torch.no_grad():
            embeddings = model(**inputs)  # (batch, num_patches, 128)
        for emb in embeddings:
            all_embeddings.append(emb.float().cpu().tolist())
    return all_embeddings


def embed_queries(
    queries: List[str],
    model: ColPali,
    processor: ColPaliProcessor,
    device: str = "cpu",
) -> List[List[List[float]]]:
    """Embed text queries with ColPali."""
    inputs = processor.process_queries(queries).to(device)
    with torch.no_grad():
        embeddings = model(**inputs)  # (batch, num_tokens, 128)
    return [emb.float().cpu().tolist() for emb in embeddings]


def setup_qdrant_collection(client: QdrantClient, recreate: bool = False):
    """Create Qdrant collection with MultiVector support for MaxSim scoring."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        if not recreate:
            print(f"Collection '{COLLECTION_NAME}' already exists.")
            return
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=COLPALI_DIM,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            ),
        ),
    )
    print(f"Created Qdrant collection: {COLLECTION_NAME}")


def ingest_pdf(
    pdf_path: str,
    model: ColPali,
    processor: ColPaliProcessor,
    client: QdrantClient,
    device: str = "cpu",
    dpi: int = 150,
) -> int:
    """
    Ingest a single PDF: convert pages → embed → upsert to Qdrant.
    Returns number of pages indexed.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    pdf_name = Path(pdf_path).stem
    print(f"\nIngesting: {pdf_name}")

    # Convert PDF to images
    images = pdf_to_images(pdf_path, dpi=dpi)
    print(f"  Pages: {len(images)}")

    # Save page images for later display in UI
    img_paths = []
    for i, img in enumerate(images):
        save_path = os.path.join(IMAGE_DIR, f"{pdf_name}_page{i+1:04d}.jpg")
        img.save(save_path, "JPEG", quality=85)
        img_paths.append(save_path)

    # Embed all pages
    print(f"  Embedding {len(images)} pages...")
    embeddings = embed_images(images, model, processor, device=device)

    # Get current max ID in collection
    collection_info = client.get_collection(COLLECTION_NAME)
    current_count = collection_info.points_count or 0

    # Upsert to Qdrant
    points = []
    for i, (emb, img_path) in enumerate(zip(embeddings, img_paths)):
        point_id = current_count + i
        points.append(
            PointStruct(
                id=point_id,
                vector=emb,  # list of patch vectors (MultiVector)
                payload={
                    "doc_name": pdf_name,
                    "pdf_path": pdf_path,
                    "page_num": i + 1,
                    "total_pages": len(images),
                    "image_path": img_path,
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  Indexed {len(points)} pages → Qdrant")

    # Save index metadata
    meta_path = os.path.join(INDEX_DIR, f"{pdf_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "pdf_path": pdf_path,
                "doc_name": pdf_name,
                "pages": len(images),
                "qdrant_ids": list(range(current_count, current_count + len(images))),
            },
            f,
            indent=2,
        )

    return len(images)


def ingest_all(pdf_dir: str = "data/pdfs", recreate: bool = False):
    """Ingest all PDFs in a directory."""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_colpali(device)
    client = QdrantClient(path="data/qdrant_db")
    setup_qdrant_collection(client, recreate=recreate)

    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        return

    total_pages = 0
    for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs"):
        pages = ingest_pdf(str(pdf_path), model, processor, client, device=device)
        total_pages += pages

    print(f"\nDone. Total pages indexed: {total_pages}")
    return client


if __name__ == "__main__":
    ingest_all(pdf_dir="data/pdfs", recreate=False)
