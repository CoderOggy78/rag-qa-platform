from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    source_type: str          # "pdf" | "txt" | "url"
    num_chunks: int
    uploaded_at: datetime
    size_bytes: int


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    message: str


class Source(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    page: Optional[int] = None
    score: float = Field(..., description="Cosine similarity 0-1")
    excerpt: str = Field(..., description="Short text snippet from chunk")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_ids: Optional[list[str]] = Field(
        default=None,
        description="Limit retrieval to specific documents. None = all.",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    model_used: str
    latency_ms: int


class DeleteResponse(BaseModel):
    doc_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    num_documents: int
    num_chunks: int
    embedding_model: str
    llm_model: str
