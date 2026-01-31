from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import pdfplumber

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_DIR,
    INDEX_DIR,
    MIN_PAGE_TEXT_LEN,
    PDF_DIR,
)
from .tokenize import tokenize

try:
    import pytesseract
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover
    pytesseract = None
    convert_from_path = None


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = " ".join(text.split())
    return text.strip()


def page_text_with_ocr(pdf_path: Path, page_index: int) -> str:
    if pytesseract is None or convert_from_path is None:
        return ""
    try:
        images = convert_from_path(str(pdf_path), first_page=page_index + 1, last_page=page_index + 1)
    except Exception:
        return ""
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


def split_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)
        if end == len(text):
            break
    return chunks


def ingest_pdf(pdf_path: Path) -> List[dict]:
    results: List[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = normalize_text(text)
            if len(text) < MIN_PAGE_TEXT_LEN:
                ocr_text = page_text_with_ocr(pdf_path, i)
                ocr_text = normalize_text(ocr_text)
                if len(ocr_text) > len(text):
                    text = ocr_text
            if not text:
                continue
            chunks = split_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for j, chunk in enumerate(chunks):
                results.append(
                    {
                        "chunk_id": f"{pdf_path.stem}-p{i+1}-c{j+1}",
                        "text": chunk,
                        "source": str(pdf_path),
                        "page": i + 1,
                        "title": pdf_path.stem,
                    }
                )
    return results


def main() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: List[dict] = []
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    for pdf in pdfs:
        chunks = ingest_pdf(pdf)
        chunk_path = CHUNKS_DIR / f"{pdf.stem}.jsonl"
        with chunk_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise SystemExit("No chunks created. Check PDFs or OCR settings.")

    texts = [c["text"] for c in all_chunks]
    tokens = [tokenize(t) for t in texts]
    with (INDEX_DIR / "bm25_tokens.json").open("w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False)

    with (INDEX_DIR / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Ingested {len(all_chunks)} chunks from {len(pdfs)} PDFs.")


if __name__ == "__main__":
    main()
