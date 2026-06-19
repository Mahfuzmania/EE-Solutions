from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import requests

from .config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
)


class Embedder(Protocol):
    provider: str
    model: str

    def embed_texts(self, texts: Sequence[str], show_progress_bar: bool = False) -> np.ndarray:
        ...

    def embed_query(self, query: str) -> np.ndarray:
        ...


class LocalSentenceTransformerEmbedder:
    provider = "local"

    def __init__(self, model: str = EMBEDDING_MODEL, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.model = model
        self.batch_size = batch_size
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from sentence_transformers import SentenceTransformer

            self._client = SentenceTransformer(self.model)
        return self._client

    def embed_texts(self, texts: Sequence[str], show_progress_bar: bool = False) -> np.ndarray:
        vectors = self.client.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query], show_progress_bar=False)[0]


class OpenAICompatibleEmbedder:
    provider = "remote"

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        base_url: str = EMBEDDING_BASE_URL,
        api_key: str = EMBEDDING_API_KEY,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        if not base_url:
            raise RuntimeError("EMBEDDING_BASE_URL is required for EMBEDDING_PROVIDER=remote")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size

    def embed_texts(self, texts: Sequence[str], show_progress_bar: bool = False) -> np.ndarray:
        del show_progress_bar
        vectors: list[list[float]] = []
        items = list(texts)
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = requests.post(
                self.base_url + "/embeddings",
                headers=headers,
                json={"model": self.model, "input": batch},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            vectors.extend(item["embedding"] for item in sorted(data, key=lambda x: x.get("index", 0)))
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query], show_progress_bar=False)[0]


def get_embedder(provider: str = EMBEDDING_PROVIDER) -> Embedder:
    provider = provider.lower()
    if provider == "local":
        return LocalSentenceTransformerEmbedder()
    if provider == "remote":
        return OpenAICompatibleEmbedder()
    raise RuntimeError(f"Unsupported embedding provider: {provider}")


_default_embedder: Embedder | None = None


def get_default_embedder() -> Embedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = get_embedder()
    return _default_embedder


# Backward-compatible helpers for existing call sites and scripts.
def embed_texts(texts: Sequence[str], show_progress_bar: bool = False) -> np.ndarray:
    return get_default_embedder().embed_texts(texts, show_progress_bar=show_progress_bar)


def embed_query(query: str) -> np.ndarray:
    return get_default_embedder().embed_query(query)
