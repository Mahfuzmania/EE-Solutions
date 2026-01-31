from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .config import INDEX_DIR, MAX_CONTEXT_CHARS, TOP_K

EMB_FILE = INDEX_DIR / "embeddings.npy"
META_FILE = INDEX_DIR / "metadata.jsonl"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int
    title: str


class VectorIndex:
    def __init__(self, embeddings: np.ndarray, chunks: List[Chunk]):
        self.embeddings = embeddings.astype(np.float32)
        self.chunks = chunks
        self._normalize()

    def _normalize(self) -> None:
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

    @classmethod
    def load(cls) -> "VectorIndex":
        if not EMB_FILE.exists() or not META_FILE.exists():
            raise FileNotFoundError("Index files not found. Run ingest.py first.")
        embeddings = np.load(EMB_FILE)
        chunks: List[Chunk] = []
        with META_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                chunks.append(
                    Chunk(
                        chunk_id=raw["chunk_id"],
                        text=raw["text"],
                        source=raw["source"],
                        page=raw["page"],
                        title=raw.get("title", ""),
                    )
                )
        return cls(embeddings, chunks)

    def search(self, query_emb: np.ndarray, top_k: int = TOP_K) -> List[Chunk]:
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        query_emb = query_emb.astype(np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9)
        scores = np.dot(self.embeddings, query_emb.T).squeeze()
        idx = np.argsort(-scores)[:top_k]
        return [self.chunks[i] for i in idx]


def build_context(chunks: Iterable[Chunk]) -> str:
    context_parts: List[str] = []
    total = 0
    for c in chunks:
        header = f"[Source: {Path(c.source).name}, page {c.page}]\n"
        block = header + c.text.strip() + "\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(block)
        total += len(block)
    return "\n".join(context_parts)
