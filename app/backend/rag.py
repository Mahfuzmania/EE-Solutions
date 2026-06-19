from __future__ import annotations

from typing import Iterable, List

from .config import TOP_K
from .embeddings import Embedder, get_default_embedder
from .prompts import build_context
from .vectorstores import Chunk, VectorStore, get_vector_store


class VectorIndex:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    @classmethod
    def load(cls) -> "VectorIndex":
        embedder = get_default_embedder()
        store = get_vector_store()
        if hasattr(store, "load"):
            store.load(embedder=embedder)  # type: ignore[attr-defined]
        return cls(store, embedder)

    def search(self, query: str, top_k: int = TOP_K) -> List[Chunk]:
        return self.store.search(query, top_k=top_k, embedder=self.embedder)


__all__ = ["Chunk", "VectorIndex", "build_context"]
