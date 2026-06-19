# Electrical Engineering RAG Bot

Local-first EE chatbot with embedding-based RAG over your textbooks. Supports English + Bengali, citations, and step-by-step answers (toggle).

## Folders
- `app/backend`: Flask backend
- `app/frontend`: static web UI
- `app/data/pdfs`: put your PDF books here
- `app/data/index`: embedding/vector index output
- `app/data/chunks`: chunked text output

## Quick start (Native)

### Linux/macOS
```bash
cd /path/to/EE-Solutions
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/backend/requirements.txt
python -m app.backend.ingest
python -m app.backend.main
```
Open `http://localhost:8000`.

### Windows (PowerShell)
```powershell
cd "C:\path\to\EE-Solutions"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app\backend\requirements.txt
python -m app.backend.ingest
python -m app.backend.main
```
Open `http://localhost:8000`.

## Quick start (Docker)

Ensure Docker is installed.

```bash
cd /path/to/EE-Solutions
docker compose up --build
```
Open `http://localhost:8000`.

To rebuild after changes:
```bash
docker compose up --build
```


## Modular RAG Providers

The RAG pipeline is split into provider modules:
- `app/backend/embeddings.py`: `local` sentence-transformers or `remote` OpenAI-compatible embeddings.
- `app/backend/vectorstores.py`: `local` NumPy vector store or `pgvector` Postgres vector store.
- `app/backend/prompts.py`: context and chat prompt construction.
- `app/backend/llms.py`: `ollama` or `remote` OpenAI-compatible chat clients.

After changing `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL`, rerun:
```bash
python -m app.backend.ingest
```
The app validates that the runtime embedding provider/model matches the generated index.


### pgvector setup

For Postgres-backed retrieval:
```bash
docker compose up -d postgres
VECTOR_STORE_PROVIDER=pgvector python -m app.backend.ingest
VECTOR_STORE_PROVIDER=pgvector python -m app.backend.main
```

When running the app inside Docker Compose, use `VECTOR_STORE_PROVIDER=pgvector` in `.env`; `DATABASE_URL` is set to the compose Postgres service automatically.

## Configuration
Create `.env` in the project root:
```
# Embeddings
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
VECTOR_STORE_PROVIDER=local
DATABASE_URL=postgresql://ee_rag:ee_rag@localhost:15432/ee_rag

# LLM
LLM_PROVIDER=ollama  # or remote
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
REMOTE_BASE_URL=http://localhost:1234/v1
REMOTE_API_KEY=
REMOTE_MODEL=gpt-4o-mini

# RAG
TOP_K=5
MAX_CONTEXT_CHARS=12000
```

## Notes
- For OCR on scanned PDFs, install Tesseract (on native) or it's included in Docker.
- For offline LLM, install Ollama and pull a model.
- For online LLM, point `REMOTE_BASE_URL` to any OpenAI-compatible endpoint.
