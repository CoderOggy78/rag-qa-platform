# RAG Document Q&A Platform

A production-grade Retrieval-Augmented Generation (RAG) system. Upload PDFs, text files, or URLs — then ask questions and get answers grounded in your documents, with cited sources.

## Architecture

```
User ──► Streamlit UI ──► FastAPI backend
                               │
                    ┌──────────┴──────────┐
                    │                     │
              Ingestion              Query pipeline
            ──────────           ──────────────────
            Parse text           Embed question
            Chunk text           ANN vector search
            Embed chunks         Build prompt + context
            Store in ChromaDB    Call LLM (OpenAI / Anthropic)
                                 Return answer + sources
```

**Stack:**
- `FastAPI` — REST + SSE streaming API
- `ChromaDB` — local persistent vector store (cosine similarity, HNSW index)
- `sentence-transformers` — local embeddings (no API key needed, `all-MiniLM-L6-v2`)
- `pdfplumber` — PDF text extraction with page numbers
- `OpenAI` / `Anthropic` — LLM generation (your choice)
- `Streamlit` — chat UI with source citation cards

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/yourname/rag-qa-platform.git
cd rag-qa-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY (or ANTHROPIC_API_KEY)
```

### 3. Start the backend

```bash
cd backend
python main.py
# API docs: http://localhost:8000/docs
```

### 4. Start the frontend

```bash
# In a second terminal
cd frontend
streamlit run app.py
# Opens at http://localhost:8501
```

---

## API reference

All endpoints are documented interactively at `http://localhost:8000/docs`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest/file` | Upload PDF, TXT, or Markdown |
| `POST` | `/ingest/url` | Ingest a public web URL |
| `POST` | `/query` | Ask a question (JSON response) |
| `GET` | `/query/stream` | Ask a question (SSE streaming) |
| `GET` | `/documents` | List all ingested documents |
| `DELETE` | `/documents/{id}` | Delete a document |
| `GET` | `/health` | Health check + stats |

### Example: ingest a file

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@report.pdf"
```

### Example: ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What were the key findings?", "top_k": 5}'
```

### Example: streaming query

```bash
curl -N "http://localhost:8000/query/stream?question=What+are+the+main+risks%3F"
```

---

## Configuration

All settings are controlled via `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for generation |
| `OPENAI_API_KEY` | — | Required if using OpenAI |
| `ANTHROPIC_API_KEY` | — | Required if using Anthropic |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Chunks to retrieve per query |
| `SIMILARITY_THRESHOLD` | `0.3` | Min cosine similarity to include |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Local vector store path |

---

## Project structure

```
rag-qa-platform/
├── backend/
│   ├── main.py          # FastAPI app and all routes
│   ├── ingestion.py     # Parse → chunk → embed → store
│   ├── retrieval.py     # Embed query → search → generate
│   ├── models.py        # Pydantic request/response schemas
│   └── config.py        # Settings via pydantic-settings
├── frontend/
│   └── app.py           # Streamlit chat UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## Extending the project

**Add more document types:**
Extend `ingestion.py` → `extract_text_from_*` with handlers for DOCX (`python-docx`), HTML, EPUB, etc.

**Swap the vector store:**
Replace ChromaDB with Qdrant, Weaviate, or pgvector by swapping `get_collection()` in `ingestion.py` and the query call in `retrieval.py`.

**Add authentication:**
Add `fastapi-users` or a simple API key middleware to `main.py`.

**Deploy:**
- Backend: `Dockerfile` with `uvicorn main:app --host 0.0.0.0 --port 8000`
- ChromaDB: mount a persistent volume at `CHROMA_PERSIST_DIR`
- Frontend: deploy on Streamlit Community Cloud or containerize alongside the API

**Production vector store:**
For > 100k chunks, switch `chromadb.PersistentClient` to a managed Qdrant or Pinecone instance and update the collection query interface in `retrieval.py`.

---

## License

MIT
