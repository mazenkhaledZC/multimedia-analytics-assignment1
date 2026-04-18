"""
Retrieval: embed a text query with ColPali → MaxSim search in Qdrant.
ColPali's late-interaction scoring computes the maximum similarity between
each query token and all document patch embeddings (like ColBERT for images).
"""

from __future__ import annotations

import torch
from typing import List, Dict, Any

from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from src.ingest import COLLECTION_NAME, embed_queries


class ColPaliRetriever:
    def __init__(
        self,
        model: ColPali,
        processor: ColPaliProcessor,
        client: QdrantClient,
        device: str = "cpu",
    ):
        self.model = model
        self.processor = processor
        self.client = client
        self.device = device

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant document pages for a query.
        Returns a list of dicts with page metadata and relevance score.
        """
        query_embeddings = embed_queries(
            [query], self.model, self.processor, device=self.device
        )
        query_vec = query_embeddings[0]  # list of token vectors

        results: List[ScoredPoint] = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            {
                "score": r.score,
                "doc_name": r.payload.get("doc_name", ""),
                "pdf_path": r.payload.get("pdf_path", ""),
                "page_num": r.payload.get("page_num", 0),
                "total_pages": r.payload.get("total_pages", 0),
                "image_path": r.payload.get("image_path", ""),
                "point_id": r.id,
            }
            for r in results
        ]

    def retrieve_multi_query(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """Retrieve for multiple queries in one batch."""
        return [self.retrieve(q, top_k) for q in queries]

    def get_collection_stats(self) -> Dict[str, Any]:
        info = self.client.get_collection(COLLECTION_NAME)
        return {
            "total_pages": info.points_count,
            "collection": COLLECTION_NAME,
            "vector_size": COLPALI_DIM,
        }


COLPALI_DIM = 128


def build_retriever(qdrant_path: str = "data/qdrant_db", device: str = None) -> ColPaliRetriever:
    from src.ingest import load_colpali

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = load_colpali(device)
    client = QdrantClient(path=qdrant_path)
    return ColPaliRetriever(model, processor, client, device)
