from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import requests
from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image
import pytesseract

from .circuits import generate_circuit
from .config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    REMOTE_API_KEY,
    REMOTE_BASE_URL,
    TOP_K,
)
from .rag import VectorIndex, build_context

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")

_index: VectorIndex | None = None


def get_index() -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex.load()
    return _index


@app.get("/")
def root() -> Response:
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@app.post("/api/reindex")
def reindex() -> Response:
    global _index
    _index = VectorIndex.load()
    return jsonify({"status": "reloaded"})


@app.post("/api/retrieve")
def retrieve() -> Response:
    payload: Dict[str, Any] = request.get_json(force=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    index = get_index()
    chunks = index.search(query, top_k=payload.get("top_k", TOP_K))
    return jsonify(
        {
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
    )


@app.post("/api/chat")
def chat() -> Response:
    payload: Dict[str, Any] = request.get_json(force=True) or {}
    query = str(payload.get("query", "")).strip()
    language = payload.get("language", "en")
    show_steps = bool(payload.get("show_steps", False))
    mode = payload.get("mode", "answer")
    user_solution = str(payload.get("solution", "")).strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    index = get_index()
    chunks = index.search(query, top_k=payload.get("top_k", TOP_K))
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
    except Exception as exc:
        error = str(exc)
        response_text = fallback_answer(query, chunks, language)

    return jsonify(
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


@app.post("/api/circuit/generate")
def circuit_generate() -> Response:
    payload = request.get_json(force=True) or {}
    image_bytes = generate_circuit(payload)
    return Response(image_bytes, mimetype="image/png")


@app.post("/api/circuit/understand")
def circuit_understand() -> Response:
    return jsonify(
        {
            "status": "not_implemented",
            "message": "Circuit understanding is not implemented yet. Upload support will be added next.",
        }
    )


@app.post("/api/ocr")
def ocr_image() -> Response:
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "empty filename"}), 400
    try:
        img = Image.open(file.stream)
        text = pytesseract.image_to_string(img)
        return jsonify({"text": text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


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
        "model": "",
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
