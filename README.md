# Local RAG

A minimal, dependency-light RAG over personal documents, running entirely
against a local LLM. No vector database, no framework.

## Setup

```bash
pip install -r requirements.txt

# Pull an embedding model (this is NOT your chat model)
ollama pull bge-m3
```

Drop your files in `./documents/` (`.pdf`, `.docx`, `.txt`, `.md`).

## Run

```bash
python main.py ingest --preview      # build the index, show sample chunks
python main.py info                  # inspect what was indexed
python main.py search "your question"  # retrieval only — debug here first
python main.py ask "your question"
python main.py chat
```

## Configuration

Everything is an environment variable, no file editing required.

| Variable | Default | Notes |
|---|---|---|
| `RAG_DOCS_DIR` | `./documents` | Source files |
| `RAG_INDEX_DIR` | `./index` | Generated index |
| `RAG_EMBED_BACKEND` | `ollama` | `ollama`, `openai`, `sentence-transformers` |
| `RAG_EMBED_MODEL` | `bge-m3` | Multilingual, good on French |
| `RAG_EMBED_URL` | `http://localhost:11434/api/embed` | |
| `RAG_LLM_BACKEND` | `ollama` | `openai` for llama.cpp / vLLM / LM Studio |
| `RAG_LLM_MODEL` | `qwen2.5:7b` | |
| `RAG_LLM_URL` | `http://localhost:11434/api/chat` | |
| `RAG_CHUNK_SIZE` | `1200` | Characters (~300 tokens) |
| `RAG_CHUNK_OVERLAP` | `180` | |
| `RAG_TOP_K` | `5` | |
| `RAG_MIN_SCORE` | `0.25` | Relevance cutoff |

### Non-Ollama server

```bash
export RAG_LLM_BACKEND=openai
export RAG_LLM_URL=http://192.168.1.50:8080/v1/chat/completions
export RAG_EMBED_BACKEND=openai
export RAG_EMBED_URL=http://192.168.1.50:8080/v1/embeddings
```

### e5 embedding models

They require asymmetric prefixes:

```bash
export RAG_EMBED_MODEL=intfloat/multilingual-e5-large
export RAG_EMBED_BACKEND=sentence-transformers
export RAG_QUERY_PREFIX="query: "
export RAG_PASSAGE_PREFIX="passage: "
```

## Tuning

Diagnose with `search`, not `ask`.

| Symptom | Fix |
|---|---|
| Right passages, wrong answer | Generation problem: bigger model, or lower temperature |
| Answer is in your docs but not retrieved | Lower `RAG_MIN_SCORE`, raise `RAG_TOP_K`, or use a bigger embedding model |
| Passages are cut mid-idea | Raise `RAG_CHUNK_SIZE` |
| Irrelevant noise in context | Raise `RAG_MIN_SCORE`, lower `RAG_TOP_K` |
| Fails on names, references, codes | Raise `RAG_KEYWORD_BOOST` |

**Any change to `RAG_EMBED_MODEL` or chunking requires a full re-ingest.**
Vectors from two different models are not comparable.

## Where to go next

- **Incremental ingestion** — hash each file, re-embed only what changed.
- **Reranking** — retrieve 20, rerank with a cross-encoder, keep 5. Biggest
  single quality gain once the basics work.
- **Conversational memory** — rewrite follow-up questions into standalone
  ones before retrieving, otherwise "and for the second one?" retrieves nothing.
- **Scale** — past ~100k chunks, swap `store.py` for `sqlite-vec` or FAISS.
  Nothing else in the codebase changes.
