from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from rank_bm25 import BM25Okapi

from .config import INDEX_DIR, MAX_CONTEXT_CHARS, TOP_K
from .tokenize import tokenize

TOKENS_FILE = INDEX_DIR / "bm25_tokens.json"
META_FILE = INDEX_DIR / "metadata.jsonl"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int
    title: str


class VectorIndex:
    def __init__(self, bm25: BM25Okapi, chunks: List[Chunk]):
        self.bm25 = bm25
        self.chunks = chunks

    @classmethod
    def load(cls) -> "VectorIndex":
        if not TOKENS_FILE.exists() or not META_FILE.exists():
            raise FileNotFoundError("Index files not found. Run ingest.py first.")
        with TOKENS_FILE.open("r", encoding="utf-8") as f:
            tokens = json.load(f)
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
        bm25 = BM25Okapi(tokens)
        return cls(bm25, chunks)

    def search(self, query: str, top_k: int = TOP_K) -> List[Chunk]:
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
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
