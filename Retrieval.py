"""
retrieval.py — Embed query → ANN search → LLM generation with citations.

Supports OpenAI and Anthropic as LLM backends.
Streaming-ready: generate_answer returns full text; use generate_answer_stream
for token-by-token streaming.
"""

import logging
import time
from typing import Optional, Generator

from ingestion import get_embedder, get_collection
from config import settings
from models import Source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(
    question: str,
    top_k: int = settings.top_k,
    doc_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Embed the question and fetch the top-k most similar chunks.

    Args:
        question:  Natural-language question string.
        top_k:     Number of chunks to return.
        doc_ids:   Optional allowlist of doc IDs to search within.

    Returns:
        List of dicts with keys: text, score, metadata.
    """
    embedder = get_embedder()
    q_embedding = embedder.encode([question], show_progress_bar=False)[0].tolist()

    collection = get_collection()
    where_filter = {"doc_id": {"$in": doc_ids}} if doc_ids else None

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(top_k, collection.count() or 1),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metas, dists):
        # ChromaDB cosine distance → similarity score
        score = round(1.0 - float(dist), 4)
        if score < settings.similarity_threshold:
            continue
        chunks.append({"text": text, "meta": meta, "score": score})

    return chunks


def build_sources(chunks: list[dict]) -> list[Source]:
    return [
        Source(
            doc_id=c["meta"]["doc_id"],
            filename=c["meta"]["filename"],
            chunk_index=c["meta"]["chunk_index"],
            page=c["meta"].get("page") if c["meta"].get("page", -1) != -1 else None,
            score=c["score"],
            excerpt=c["text"][:200].replace("\n", " ").strip() + "…",
        )
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks, 1):
        page_info = f"page {c['meta']['page']}, " if c["meta"].get("page", -1) != -1 else ""
        context_parts.append(
            f"[Source {i}: {c['meta']['filename']}, {page_info}chunk {c['meta']['chunk_index']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a precise document Q&A assistant. Answer the question using ONLY the provided source excerpts.

Rules:
- Cite sources inline using [Source N] notation.
- If the answer is not in the sources, say "I don't have enough information in the provided documents to answer this."
- Be concise but complete. Use bullet points for multi-part answers.
- Do not hallucinate or use outside knowledge.

SOURCES:
{context}

QUESTION: {question}

ANSWER:"""


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def _call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(prompt: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _call_openai_stream(prompt: str) -> Generator[str, None, None]:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    with client.chat.completions.stream(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    ) as stream:
        for text in stream.text_stream:
            yield text


def _call_anthropic_stream(prompt: str) -> Generator[str, None, None]:
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    with client.messages.stream(
        model=settings.llm_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    top_k: int = settings.top_k,
    doc_ids: Optional[list[str]] = None,
) -> dict:
    """
    Full RAG pipeline: retrieve + generate.

    Returns:
        {answer, sources, model_used, latency_ms}
    """
    t0 = time.time()

    chunks = retrieve_chunks(question, top_k=top_k, doc_ids=doc_ids)
    if not chunks:
        return {
            "answer": "No relevant content found in the uploaded documents for this question.",
            "sources": [],
            "model_used": settings.llm_model,
            "latency_ms": round((time.time() - t0) * 1000),
        }

    prompt = build_prompt(question, chunks)

    if settings.llm_provider == "anthropic":
        answer = _call_anthropic(prompt)
    else:
        answer = _call_openai(prompt)

    sources = build_sources(chunks)
    latency_ms = round((time.time() - t0) * 1000)

    logger.info(
        "Q: %.60s… | %d sources | %dms | model=%s",
        question, len(sources), latency_ms, settings.llm_model,
    )

    return {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "model_used": settings.llm_model,
        "latency_ms": latency_ms,
    }


def generate_answer_stream(
    question: str,
    top_k: int = settings.top_k,
    doc_ids: Optional[list[str]] = None,
) -> Generator[str, None, None]:
    """
    Streaming version. Yields text tokens, then a final JSON block with sources.
    Format:
        token token token ... \n\n__SOURCES__\n<json>
    """
    import json

    chunks = retrieve_chunks(question, top_k=top_k, doc_ids=doc_ids)
    if not chunks:
        yield "No relevant content found in the uploaded documents for this question."
        yield "\n\n__SOURCES__\n[]"
        return

    prompt = build_prompt(question, chunks)
    sources = build_sources(chunks)

    if settings.llm_provider == "anthropic":
        token_gen = _call_anthropic_stream(prompt)
    else:
        token_gen = _call_openai_stream(prompt)

    for token in token_gen:
        yield token

    yield f"\n\n__SOURCES__\n{json.dumps([s.model_dump() for s in sources])}"
