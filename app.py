
import json
import time
from pathlib import Path

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #4c84ff;
        border-radius: 4px;
        padding: 0.6rem 0.8rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .source-score {
        font-size: 0.75rem;
        color: #6c757d;
        font-weight: 600;
    }
    .source-excerpt {
        color: #495057;
        margin-top: 0.3rem;
    }
    .metric-box {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path: str) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is the API running on port 8000?")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def api_post(path: str, **kwargs) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend.")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def api_delete(path: str) -> dict | None:
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def render_sources(sources: list[dict]):
    if not sources:
        return
    st.markdown("**Sources cited:**")
    for i, src in enumerate(sources, 1):
        page_info = f" · page {src['page']}" if src.get("page") else ""
        score_pct = round(src["score"] * 100)
        st.markdown(
            f"""<div class="source-card">
                <span class="source-score">[{i}] {src['filename']}{page_info} · {score_pct}% match</span>
                <div class="source-excerpt">{src['excerpt']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

with st.sidebar:
    st.title("📚 RAG Q&A")

    # Health check
    health = api_get("/health")
    if health:
        col1, col2 = st.columns(2)
        col1.metric("Documents", health.get("num_documents", 0))
        col2.metric("Chunks", health.get("num_chunks", 0))
        st.caption(f"Model: `{health.get('llm_model', '—')}`")
    st.divider()

    # Upload
    st.subheader("Upload documents")
    upload_tab, url_tab = st.tabs(["File", "URL"])

    with upload_tab:
        uploaded = st.file_uploader(
            "PDF, TXT, or Markdown",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if st.button("Ingest files", disabled=not uploaded):
            for f in uploaded:
                with st.spinner(f"Indexing {f.name}…"):
                    result = api_post(
                        "/ingest/file",
                        files={"file": (f.name, f.getvalue(), f.type)},
                    )
                    if result:
                        st.success(f"✓ {f.name} — {result['num_chunks']} chunks")
            st.rerun()

    with url_tab:
        url_input = st.text_input("Public URL", placeholder="https://example.com/report.html")
        url_title = st.text_input("Title (optional)")
        if st.button("Ingest URL", disabled=not url_input):
            with st.spinner("Fetching and indexing…"):
                result = api_post(
                    "/ingest/url",
                    json={"url": url_input, "title": url_title or None},
                )
                if result:
                    st.success(f"✓ {result['num_chunks']} chunks indexed")
            st.rerun()

    st.divider()

    # Document list
    st.subheader("Indexed documents")
    docs = api_get("/documents") or []
    if not docs:
        st.caption("No documents yet. Upload something above.")
    else:
        for doc in docs:
            col_name, col_del = st.columns([4, 1])
            col_name.markdown(
                f"**{doc['filename'][:28]}{'…' if len(doc['filename']) > 28 else ''}**  \n"
                f"<span style='font-size:0.75rem;color:#6c757d'>{doc['num_chunks']} chunks · {doc['source_type']}</span>",
                unsafe_allow_html=True,
            )
            if col_del.button("🗑", key=f"del_{doc['doc_id']}", help="Delete"):
                api_delete(f"/documents/{doc['doc_id']}")
                st.rerun()


# ---------------------------------------------------------------------------
# Main area — Chat interface
# ---------------------------------------------------------------------------

st.title("Ask your documents")
st.caption("Questions are answered using only the content you've uploaded. Sources are cited.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                render_sources(msg["sources"])
        if msg.get("latency_ms"):
            st.caption(f"⏱ {msg['latency_ms']}ms · {msg.get('model_used', '')}")

# Input
if question := st.chat_input("Ask a question about your documents…"):
    if not docs:
        st.warning("Upload at least one document before asking questions.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        placeholder = st.empty()
        sources_placeholder = st.empty()
        latency_placeholder = st.empty()

        with st.spinner("Searching and generating…"):
            t0 = time.time()
            result = api_post(
                "/query",
                json={"question": question, "top_k": 5},
            )

        if result:
            answer = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])
            latency_ms = result.get("latency_ms", round((time.time() - t0) * 1000))
            model_used = result.get("model_used", "")

            placeholder.markdown(answer)
            with sources_placeholder.expander("Sources", expanded=True):
                render_sources(sources)
            latency_placeholder.caption(f"⏱ {latency_ms}ms · {model_used}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "latency_ms": latency_ms,
                    "model_used": model_used,
                }
            )
        else:
            placeholder.error("Failed to get an answer. Check the backend logs.")

# Clear chat button
if st.session_state.messages:
    if st.button("Clear chat history", type="secondary"):
        st.session_state.messages = []
        st.rerun()
