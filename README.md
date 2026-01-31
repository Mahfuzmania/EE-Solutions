# Electrical Engineering RAG Bot

Local-first EE chatbot with RAG over your textbooks. Supports English + Bengali, citations, and step-by-step answers (toggle).

## Folders
- `app/backend`: FastAPI backend
- `app/frontend`: static web UI
- `app/data/pdfs`: put your PDF books here
- `app/data/index`: vector index output
- `app/data/chunks`: chunked text output

## Quick start (Windows PowerShell)
```powershell
cd "f:\AI Agent"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app\backend\requirements.txt
python app\backend\ingest.py
uvicorn app.backend.main:app --reload --port 8000
```
Open `http://localhost:8000`.

## Configuration
Create `.env` in `f:\AI Agent`:
```
# Embeddings
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

# LLM
LLM_PROVIDER=ollama  # or remote
OLLAMA_MODEL=llama3.1:8b
REMOTE_BASE_URL=http://localhost:1234/v1
REMOTE_API_KEY=

# RAG
TOP_K=5
MAX_CONTEXT_CHARS=12000
```

## Notes
- For OCR on scanned PDFs, install Tesseract and add it to PATH.
- For offline LLM, install Ollama and pull a model.
- For online LLM, point `REMOTE_BASE_URL` to any OpenAI-compatible endpoint.
