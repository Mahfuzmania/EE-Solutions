from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image
import pytesseract

from .circuits import generate_circuit
from .config import (
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    REMOTE_BASE_URL,
    REMOTE_MODEL,
    TOP_K,
    VECTOR_STORE_PROVIDER,
)
from .llms import get_llm_client
from .prompts import build_chat_prompt
from .rag import VectorIndex

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")

# Ensure Tesseract is found even if PATH is not refreshed in the session.
_tesseract_default = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
if _tesseract_default.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_tesseract_default)

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


@app.get("/api/config")
def config_status() -> Response:
    return jsonify(
        {
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_base_url": EMBEDDING_BASE_URL,
            "vector_store_provider": VECTOR_STORE_PROVIDER,
            "llm_provider": LLM_PROVIDER,
            "remote_base_url": REMOTE_BASE_URL,
            "remote_model": REMOTE_MODEL,
            "ollama_base_url": OLLAMA_BASE_URL,
            "ollama_model": OLLAMA_MODEL,
            "ocr_ready": bool(getattr(pytesseract.pytesseract, "tesseract_cmd", "")),
        }
    )


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
    system, prompt = build_chat_prompt(
        query=query,
        chunks=chunks,
        language=language,
        show_steps=show_steps,
        mode=mode,
        user_solution=user_solution,
    )

    response_text = None
    error = None

    try:
        response_text = get_llm_client().chat(system, prompt)
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



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
