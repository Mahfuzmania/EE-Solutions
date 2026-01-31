from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    REMOTE_API_KEY,
    REMOTE_BASE_URL,
    TOP_K,
)
from .embeddings import Embedder
from .rag import VectorIndex, build_context

app = FastAPI(title="EE RAG Bot")

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

_embedder: Embedder | None = None
_index: VectorIndex | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_index() -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex.load()
    return _index


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/reindex")
def reindex() -> Dict[str, str]:
    global _index
    _index = VectorIndex.load()
    return {"status": "reloaded"}


@app.post("/api/retrieve")
def retrieve(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    embedder = get_embedder()
    index = get_index()
    q_emb = embedder.embed([query])[0]
    chunks = index.search(q_emb, top_k=payload.get("top_k", TOP_K))
    return {
        "results": [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "page": c.page,
                "title": c.title,
                "text": c.text,
            }
            for c in chunks
        ]
    }


@app.post("/api/chat")
def chat(payload: Dict[str, Any]) -> JSONResponse:
    query = payload.get("query", "").strip()
    language = payload.get("language", "en")
    show_steps = bool(payload.get("show_steps", False))
    mode = payload.get("mode", "answer")
    user_solution = payload.get("solution", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    embedder = get_embedder()
    index = get_index()
    q_emb = embedder.embed([query])[0]
    chunks = index.search(q_emb, top_k=payload.get("top_k", TOP_K))
    context = build_context(chunks)

    system = (
        "You are an expert Electrical & Electronic Engineering tutor. "
        "Always cite sources using the [Source: file, page] format. "
        "Be precise with equations, units, and steps. "
        "If the user asks to check a solution, identify errors and provide corrections."
    )

    instructions = [
        f"Language: {'Bengali' if language == 'bn' else 'English'}.",
        "Use the provided context for factual claims.",
        "If something is not in context, say it is not found in the sources.",
        "Provide step-by-step reasoning only if show_steps is true.",
    ]

    if mode == "check":
        instructions.append("The user provided a solution. Evaluate and correct it.")

    user_prompt = query
    if user_solution:
        user_prompt += f"\n\nUser solution:\n{user_solution}"

    prompt = (
        "CONTEXT:\n"
        f"{context}\n\n"
        "INSTRUCTIONS:\n"
        + "\n".join(instructions)
        + "\n\nUSER QUESTION:\n"
        + user_prompt
    )

    response_text = None
    error = None

    try:
        if LLM_PROVIDER == "ollama":
            response_text = ollama_chat(system, prompt)
        elif LLM_PROVIDER == "remote":
            response_text = remote_chat(system, prompt)
        else:
            raise RuntimeError("LLM provider not configured")
    except Exception as exc:  # fallback
        error = str(exc)
        response_text = fallback_answer(query, chunks, language)

    return JSONResponse(
        {
            "answer": response_text,
            "sources": [
                {
                    "source": c.source,
                    "page": c.page,
                    "title": c.title,
                    "chunk_id": c.chunk_id,
                }
                for c in chunks
            ],
            "error": error,
        }
    )


def fallback_answer(query: str, chunks: List[Any], language: str) -> str:
    if language == "bn":
        header = "LLM ??????? ??? ???? ???? ?????? ?????????? ??????? ????? ??:"
    else:
        header = "LLM is not configured. Here are the most relevant sources:"
    lines = [header, ""]
    for c in chunks:
        lines.append(f"[Source: {Path(c.source).name}, page {c.page}] {c.text[:400]}")
    return "\n".join(lines)


def ollama_chat(system: str, prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def remote_chat(system: str, prompt: str) -> str:
    url = REMOTE_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if REMOTE_API_KEY:
        headers["Authorization"] = f"Bearer {REMOTE_API_KEY}"
    payload = {
        "model": "",  # set in remote server default
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
