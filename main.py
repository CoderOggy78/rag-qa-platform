"""
main.py — FastAPI application entry point.

Endpoints:
    POST   /ingest/file     Upload a PDF or TXT file
    POST   /ingest/url      Ingest a web URL
    POST   /query           Ask a question (non-streaming)
    GET    /query/stream    Ask a question (SSE streaming)
    GET    /documents       List all ingested documents
    DELETE /documents/{id}  Remove a document
    GET    /health          Health check
"""

import logging
import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl

from config import settings, UPLOAD_DIR
from ingestion import ingest_document, delete_document, list_documents, collection_stats
from retrieval import generate_answer, generate_answer_stream
from models import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    DeleteResponse,
    HealthResponse,
    Source,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Q&A Platform",
    description="Upload documents, ask questions, get cited answers.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """Upload a PDF, TXT, or Markdown file for indexing."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    doc_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{doc_id}{suffix}"
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    source_type = "pdf" if suffix == ".pdf" else "txt"
    try:
        meta = ingest_document(
            source=dest,
            filename=file.filename or dest.name,
            source_type=source_type,
            doc_id=doc_id,
        )
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=422, detail=str(exc))

    return IngestResponse(
        doc_id=meta["doc_id"],
        filename=meta["filename"],
        num_chunks=meta["num_chunks"],
        message=f"Successfully indexed {meta['num_chunks']} chunks.",
    )


class URLIngestRequest(BaseModel):
    url: HttpUrl
    title: Optional[str] = None


@app.post("/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_url(body: URLIngestRequest):
    """Ingest a public web page by URL."""
    url_str = str(body.url)
    filename = body.title or url_str[:80]
    try:
        meta = ingest_document(
            source=url_str,
            filename=filename,
            source_type="url",
        )
    except Exception as exc:
        logger.exception("URL ingestion failed: %s", url_str)
        raise HTTPException(status_code=422, detail=str(exc))

    return IngestResponse(
        doc_id=meta["doc_id"],
        filename=meta["filename"],
        num_chunks=meta["num_chunks"],
        message=f"Successfully indexed {meta['num_chunks']} chunks from URL.",
    )

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(body: QueryRequest):
    """Ask a question. Returns answer + cited sources."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = generate_answer(
            question=body.question,
            top_k=body.top_k,
            doc_ids=body.doc_ids,
        )
    except Exception as exc:
        logger.exception("Query failed: %s", body.question)
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        model_used=result["model_used"],
        latency_ms=result["latency_ms"],
    )


@app.get("/query/stream", tags=["Query"])
async def query_stream(
    question: str = Query(..., min_length=3),
    top_k: int = Query(default=5, ge=1, le=20),
    doc_ids: Optional[str] = Query(default=None, description="Comma-separated doc IDs"),
):
    """
    Ask a question with Server-Sent Events streaming.
    Each SSE event is a text token. Final event contains JSON sources block.
    """
    doc_id_list = [d.strip() for d in doc_ids.split(",")] if doc_ids else None

    def event_generator():
        try:
            for token in generate_answer_stream(question, top_k=top_k, doc_ids=doc_id_list):
                # SSE format: data: <payload>\n\n
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            logger.exception("Streaming query error")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

@app.get("/documents", tags=["Documents"])
async def list_docs():
    """List all ingested documents with their metadata."""
    return list_documents()


@app.delete("/documents/{doc_id}", response_model=DeleteResponse, tags=["Documents"])
async def delete_doc(doc_id: str):
    """Remove a document and all its chunks from the vector store."""
    deleted = delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    # Also remove uploaded file if it exists
    for f in UPLOAD_DIR.glob(f"{doc_id}.*"):
        f.unlink(missing_ok=True)
    return DeleteResponse(doc_id=doc_id, message=f"Deleted {deleted} chunks.")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check with vector store statistics."""
    stats = collection_stats()
    return HealthResponse(
        status="ok",
        num_documents=stats["num_documents"],
        num_chunks=stats["num_chunks"],
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
    )


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
