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
| `RAG_CHUNK_SIZE` | `500` | Characters. Tuned for Japanese — use `1200` for Latin scripts |
| `RAG_CHUNK_OVERLAP` | `80` | Use `180` for Latin scripts |
| `RAG_TOP_K` | `5` | |
| `RAG_MIN_SCORE` | `0.25` | Relevance cutoff — **model-dependent, see below** |

### Non-Ollama server

```bash
export RAG_LLM_BACKEND=openai
export RAG_LLM_URL=http://192.168.1.50:8080/v1/chat/completions
export RAG_EMBED_BACKEND=openai
export RAG_EMBED_URL=http://192.168.1.50:8080/v1/embeddings
```

## Japanese

The loader, chunker and keyword matcher all handle Japanese:

- sentences split on `。！？` with no whitespace, keeping `「…」` attached
- hard-wrapped lines from Japanese PDFs are rejoined
- no spaces are inserted when reassembling sentences
- chunk overlap restarts at a sentence boundary
- keyword matching uses character bigrams, since `\w+` would swallow a whole
  clause as a single token and never match

### Embedding model

`bge-m3` is multilingual and decent on Japanese — the zero-setup default.
Japanese-specialized models score higher on JMTEB:

```bash
pip install sentence-transformers
export RAG_EMBED_BACKEND=sentence-transformers
export RAG_EMBED_MODEL=cl-nagoya/ruri-v3-310m
```

**Ruri requires Japanese prefixes.** Check the exact strings on the model
card and set `RAG_QUERY_PREFIX` / `RAG_PASSAGE_PREFIX` accordingly —
omitting them fails silently and costs real accuracy.

For the e5 family:

```bash
export RAG_EMBED_MODEL=intfloat/multilingual-e5-large
export RAG_EMBED_BACKEND=sentence-transformers
export RAG_QUERY_PREFIX="query: "
export RAG_PASSAGE_PREFIX="passage: "
```

### Calibrate min_score per model

Similarity scores are not comparable across models. e5 and ruri push most
pairs into the 0.7–0.9 range, so the default `0.25` lets everything through
and the threshold does nothing.

Run `search` on a question you know is **not** covered by your documents,
look at the top score, and set `RAG_MIN_SCORE` just above it.

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
