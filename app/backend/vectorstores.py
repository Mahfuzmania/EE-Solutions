from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .config import DATABASE_URL, INDEX_DIR, VECTOR_STORE_PROVIDER
from .embeddings import Embedder, get_default_embedder

INDEX_ID = "default"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int
    title: str


class VectorStore(Protocol):
    provider: str

    def search(self, query: str, top_k: int, embedder: Embedder | None = None) -> list[Chunk]:
        ...

    def save(self, chunks: Sequence[dict], embedder: Embedder, show_progress_bar: bool = False) -> None:
        ...


class LocalNpyVectorStore:
    provider = "local"

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.meta_file = index_dir / "metadata.jsonl"
        self.embeddings_file = index_dir / "embeddings.npy"
        self.config_file = index_dir / "embedding_config.json"
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.embedding_config: dict = {}

    def load(self, embedder: Embedder | None = None) -> "LocalNpyVectorStore":
        if not self.embeddings_file.exists() or not self.meta_file.exists():
            raise FileNotFoundError("Index files not found. Run ingest.py first.")
        self.chunks = self._read_chunks()
        self.embeddings = np.load(self.embeddings_file).astype(np.float32, copy=False)
        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(
                f"Index mismatch: {len(self.chunks)} metadata rows for {self.embeddings.shape[0]} embeddings"
            )
        if self.config_file.exists():
            self.embedding_config = json.loads(self.config_file.read_text(encoding="utf-8"))
            if embedder is not None:
                self._validate_embedder(embedder)
        return self

    def save(self, chunks: Sequence[dict], embedder: Embedder, show_progress_bar: bool = False) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed_texts(texts, show_progress_bar=show_progress_bar)
        with self.meta_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        with self.embeddings_file.open("wb") as f:
            np.save(f, embeddings)
        config = {
            "provider": embedder.provider,
            "model": embedder.model,
            "dimensions": int(embeddings.shape[1]),
            "normalized": True,
            "chunks": len(chunks),
            "vector_store_provider": self.provider,
        }
        self.config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        self.chunks = [self._chunk_from_dict(chunk) for chunk in chunks]
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.embedding_config = config

    def search(self, query: str, top_k: int, embedder: Embedder | None = None) -> list[Chunk]:
        if self.embeddings is None:
            self.load(embedder=embedder)
        embedder = embedder or get_default_embedder()
        self._validate_embedder(embedder)
        assert self.embeddings is not None
        query_embedding = embedder.embed_query(query)
        scores = self.embeddings @ query_embedding
        limit = max(1, min(int(top_k), len(self.chunks)))
        idx = np.argsort(-scores)[:limit]
        return [self.chunks[i] for i in idx]

    def _read_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        with self.meta_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(self._chunk_from_dict(json.loads(line)))
        return chunks

    def _validate_embedder(self, embedder: Embedder) -> None:
        if not self.embedding_config:
            return
        _validate_embedding_config(self.embedding_config, embedder)

    @staticmethod
    def _chunk_from_dict(raw: dict) -> Chunk:
        return Chunk(
            chunk_id=raw["chunk_id"],
            text=raw["text"],
            source=raw["source"],
            page=raw["page"],
            title=raw.get("title", ""),
        )


class PgVectorStore:
    provider = "pgvector"

    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self.embedding_config: dict = {}

    def load(self, embedder: Embedder | None = None) -> "PgVectorStore":
        self._ensure_schema()
        self.embedding_config = self._read_embedding_config()
        if self.embedding_config and embedder is not None:
            _validate_embedding_config(self.embedding_config, embedder)
        return self

    def save(self, chunks: Sequence[dict], embedder: Embedder, show_progress_bar: bool = False) -> None:
        self._ensure_schema()
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed_texts(texts, show_progress_bar=show_progress_bar)
        dimensions = int(embeddings.shape[1])
        self._ensure_embedding_dimension(dimensions)

        import psycopg

        rows = [
            (
                chunk["chunk_id"],
                chunk["text"],
                chunk["source"],
                int(chunk["page"]),
                chunk.get("title", ""),
                _to_pgvector(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        config = {
            "provider": embedder.provider,
            "model": embedder.model,
            "dimensions": dimensions,
            "normalized": True,
            "chunks": len(chunks),
            "vector_store_provider": self.provider,
        }

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE rag_chunks")
                cur.executemany(
                    """
                    INSERT INTO rag_chunks (chunk_id, text, source, page, title, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    """,
                    rows,
                )
                cur.execute(
                    """
                    INSERT INTO rag_index_metadata (
                        id, embedding_provider, embedding_model, dimensions,
                        vector_store_provider, chunk_count, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET
                        embedding_provider = EXCLUDED.embedding_provider,
                        embedding_model = EXCLUDED.embedding_model,
                        dimensions = EXCLUDED.dimensions,
                        vector_store_provider = EXCLUDED.vector_store_provider,
                        chunk_count = EXCLUDED.chunk_count,
                        updated_at = now()
                    """,
                    (INDEX_ID, embedder.provider, embedder.model, dimensions, self.provider, len(chunks)),
                )
        self.embedding_config = config

    def search(self, query: str, top_k: int, embedder: Embedder | None = None) -> list[Chunk]:
        embedder = embedder or get_default_embedder()
        if not self.embedding_config:
            self.load(embedder=embedder)
        _validate_embedding_config(self.embedding_config, embedder)
        query_embedding = embedder.embed_query(query)
        limit = max(1, int(top_k))

        import psycopg

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, text, source, page, title
                    FROM rag_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (_to_pgvector(query_embedding), limit),
                )
                return [
                    Chunk(chunk_id=row[0], text=row[1], source=row[2], page=row[3], title=row[4])
                    for row in cur.fetchall()
                ]

    def _ensure_schema(self) -> None:
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_index_metadata (
                        id text PRIMARY KEY,
                        embedding_provider text NOT NULL,
                        embedding_model text NOT NULL,
                        dimensions integer NOT NULL,
                        vector_store_provider text NOT NULL,
                        chunk_count integer NOT NULL,
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        id bigserial PRIMARY KEY,
                        chunk_id text NOT NULL UNIQUE,
                        text text NOT NULL,
                        source text NOT NULL,
                        page integer NOT NULL,
                        title text NOT NULL,
                        embedding vector
                    )
                    """
                )

    def _ensure_embedding_dimension(self, dimensions: int) -> None:
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE rag_chunks")
                cur.execute("ALTER TABLE rag_chunks DROP COLUMN IF EXISTS embedding")
                cur.execute(f"ALTER TABLE rag_chunks ADD COLUMN embedding vector({dimensions}) NOT NULL")

    def _read_embedding_config(self) -> dict:
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT embedding_provider, embedding_model, dimensions,
                           vector_store_provider, chunk_count
                    FROM rag_index_metadata
                    WHERE id = %s
                    """,
                    (INDEX_ID,),
                )
                row = cur.fetchone()
        if row is None:
            raise FileNotFoundError("pgvector index not found. Run ingest.py first.")
        return {
            "provider": row[0],
            "model": row[1],
            "dimensions": row[2],
            "vector_store_provider": row[3],
            "chunks": row[4],
            "normalized": True,
        }


def _validate_embedding_config(config: dict, embedder: Embedder) -> None:
    expected_provider = config.get("provider")
    expected_model = config.get("model")
    if expected_provider != embedder.provider or expected_model != embedder.model:
        raise ValueError(
            "Embedding index was built with "
            f"{expected_provider}/{expected_model}, but runtime is configured for "
            f"{embedder.provider}/{embedder.model}. Re-run ingest.py after changing embedding providers or models."
        )


def _to_pgvector(vector: np.ndarray) -> str:
    return "[" + ",".join(str(float(value)) for value in vector.tolist()) + "]"


def get_vector_store(provider: str = VECTOR_STORE_PROVIDER) -> VectorStore:
    provider = provider.lower()
    if provider == "local":
        return LocalNpyVectorStore()
    if provider == "pgvector":
        return PgVectorStore()
    raise RuntimeError(f"Unsupported vector store provider: {provider}")
