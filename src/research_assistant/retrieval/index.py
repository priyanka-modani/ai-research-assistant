import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from research_assistant.config import Settings
from research_assistant.providers import ModelProvider
from research_assistant.retrieval.fusion import reciprocal_rank_fusion
from research_assistant.schemas import DocumentChunk, RetrievedChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridIndex:
    def __init__(self, settings: Settings, provider: ModelProvider) -> None:
        self.settings = settings
        self.provider = provider
        self.chunks = self._load_chunks(settings.chunks_path)
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks])
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = client.get_collection(settings.chroma_collection)

    @staticmethod
    def _load_chunks(path: Path) -> list[DocumentChunk]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [DocumentChunk.model_validate(item) for item in payload]

    def dense(self, query: str) -> list[RetrievedChunk]:
        vector = self.provider.embed([query])[0]
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(self.settings.top_k_dense, len(self.chunks)),
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [
            RetrievedChunk(
                **self.by_id[chunk_id].model_dump(),
                dense_score=1.0 / (1.0 + float(distance)),
            )
            for chunk_id, distance in zip(ids, distances, strict=True)
        ]

    def sparse(self, query: str) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[
            : self.settings.top_k_sparse
        ]
        return [
            RetrievedChunk(**self.chunks[index].model_dump(), sparse_score=float(scores[index]))
            for index in ranked
            if scores[index] > 0
        ]

    def search(self, query: str) -> list[RetrievedChunk]:
        dense = self.dense(query)
        sparse = self.sparse(query)
        fused = reciprocal_rank_fusion(
            [[item.chunk_id for item in dense], [item.chunk_id for item in sparse]]
        )
        dense_map = {item.chunk_id: item.dense_score for item in dense}
        sparse_map = {item.chunk_id: item.sparse_score for item in sparse}
        candidates = [
            RetrievedChunk(
                **self.by_id[chunk_id].model_dump(),
                dense_score=dense_map.get(chunk_id),
                sparse_score=sparse_map.get(chunk_id),
                fusion_score=score,
            )
            for chunk_id, score in fused[: max(self.settings.top_k_final * 2, 8)]
        ]
        if self.settings.enable_rerank and candidates:
            reranked = self.provider.rerank(query, [item.text for item in candidates])
            candidates = [
                candidates[index].model_copy(update={"rerank_score": score})
                for index, score in reranked
            ]
        return candidates[: self.settings.top_k_final]
