"""
ingestion.py — Parse → Chunk → Embed → Store pipeline.

Supports: PDF, plain text, URLs.
Chunking: recursive character split with overlap.
Embeddings: sentence-transformers (local, no API key needed).
Store: ChromaDB with persistent local storage.
"""

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import re

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import settings, UPLOAD_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons (loaded once at startup)
# ---------------------------------------------------------------------------

_embedder: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def get_collection():
    global _chroma_client, _collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(path: Path) -> list[dict]:
    """Return list of {page: int, text: str}."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": i + 1, "text": text})
        return pages
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")


def extract_text_from_url(url: str) -> list[dict]:
    """Fetch a URL and extract readable text."""
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=15, headers={"User-Agent": "RAG-QA/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return [{"page": None, "text": text}]
    except ImportError:
        raise RuntimeError("requests/beautifulsoup4 not installed.")


def extract_text_from_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{"page": None, "text": text}]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive character split: paragraph → sentence → word fallback."""
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks, current = [], ""
            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # carry-over overlap
                    overlap_text = current[-overlap:] if overlap else ""
                    current = overlap_text + (sep if overlap_text else "") + part
            if current:
                chunks.append(current)
            return [c.strip() for c in chunks if c.strip()]

    # Hard split fallback
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[dict]:
    """Convert page-level text into chunk dicts with metadata."""
    chunks = []
    idx = 0
    for page_data in pages:
        for chunk_text in _split_text(page_data["text"], chunk_size, overlap):
            if len(chunk_text) < 30:  # skip trivially short chunks
                continue
            chunks.append(
                {
                    "chunk_index": idx,
                    "text": chunk_text,
                    "page": page_data.get("page"),
                }
            )
            idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Ingestion entry point
# ---------------------------------------------------------------------------

def ingest_document(
    source: str | Path,
    filename: str,
    source_type: str,
    doc_id: Optional[str] = None,
) -> dict:
    """
    Full pipeline: parse → chunk → embed → upsert to ChromaDB.

    Returns metadata dict with doc_id, num_chunks, etc.
    """
    t0 = time.time()
    doc_id = doc_id or str(uuid.uuid4())

    # 1. Extract text
    if source_type == "pdf":
        pages = extract_text_from_pdf(Path(source))
    elif source_type == "url":
        pages = extract_text_from_url(str(source))
    elif source_type in ("txt", "text"):
        pages = extract_text_from_txt(Path(source))
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    if not pages:
        raise ValueError("No text could be extracted from the document.")

    # 2. Chunk
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("Document produced no usable chunks after splitting.")

    # 3. Embed
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    # 4. Build ChromaDB records
    collection = get_collection()
    ids = [f"{doc_id}__chunk_{c['chunk_index']}" for c in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "source_type": source_type,
            "chunk_index": c["chunk_index"],
            "page": c["page"] if c["page"] is not None else -1,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        for c in chunks
    ]

    # Upsert in batches of 500 (ChromaDB recommendation)
    batch = 500
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )

    elapsed = round((time.time() - t0) * 1000)
    logger.info("Ingested %d chunks from '%s' in %dms", len(chunks), filename, elapsed)

    size_bytes = 0
    if source_type in ("pdf", "txt", "text"):
        p = Path(source)
        if p.exists():
            size_bytes = p.stat().st_size

    return {
        "doc_id": doc_id,
        "filename": filename,
        "source_type": source_type,
        "num_chunks": len(chunks),
        "uploaded_at": datetime.now(timezone.utc),
        "size_bytes": size_bytes,
        "ingestion_ms": elapsed,
    }


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_document(doc_id: str) -> int:
    """Remove all chunks belonging to doc_id. Returns number of chunks deleted."""
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id})
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


# ---------------------------------------------------------------------------
# Listing / stats
# ---------------------------------------------------------------------------

def list_documents() -> list[dict]:
    """Return one summary record per unique doc_id."""
    collection = get_collection()
    all_meta = collection.get(include=["metadatas"])["metadatas"] or []
    docs: dict[str, dict] = {}
    for m in all_meta:
        did = m["doc_id"]
        if did not in docs:
            docs[did] = {
                "doc_id": did,
                "filename": m["filename"],
                "source_type": m["source_type"],
                "uploaded_at": m.get("uploaded_at", ""),
                "num_chunks": 0,
                "size_bytes": 0,
            }
        docs[did]["num_chunks"] += 1
    return list(docs.values())


def collection_stats() -> dict:
    collection = get_collection()
    count = collection.count()
    docs = list_documents()
    return {
        "num_chunks": count,
        "num_documents": len(docs),
    }
