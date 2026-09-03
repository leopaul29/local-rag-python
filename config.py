"""Central configuration for the local RAG.

Every value can be overridden with an environment variable, so you never
have to edit this file to point at a different model or server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Config:
    # --- Paths -------------------------------------------------------------
    docs_dir: Path = Path(_env("RAG_DOCS_DIR", "./documents"))
    index_dir: Path = Path(_env("RAG_INDEX_DIR", "./index"))

    # --- Embedding model ---------------------------------------------------
    # IMPORTANT: this is NOT your chat model. It must be a dedicated
    # embedding model, otherwise retrieval quality collapses.
    # backend: "ollama" | "openai" | "sentence-transformers"
    embed_backend: str = _env("RAG_EMBED_BACKEND", "ollama")
    embed_model: str = _env("RAG_EMBED_MODEL", "bge-m3")
    embed_url: str = _env("RAG_EMBED_URL", "http://localhost:11434/api/embed")
    embed_batch_size: int = _env_int("RAG_EMBED_BATCH", 16)

    # Some models (e5 family) expect asymmetric prefixes. bge-m3 does not.
    # For intfloat/multilingual-e5-*, set "query: " and "passage: ".
    query_prefix: str = _env("RAG_QUERY_PREFIX", "")
    passage_prefix: str = _env("RAG_PASSAGE_PREFIX", "")

    # --- Chat model --------------------------------------------------------
    # backend: "ollama" | "openai" (openai = any OpenAI-compatible server)
    llm_backend: str = _env("RAG_LLM_BACKEND", "ollama")
    llm_model: str = _env("RAG_LLM_MODEL", "qwen2.5:7b")
    llm_url: str = _env("RAG_LLM_URL", "http://localhost:11434/api/chat")
    llm_api_key: str = _env("RAG_LLM_API_KEY", "")
    llm_temperature: float = _env_float("RAG_LLM_TEMPERATURE", 0.1)
    request_timeout: int = _env_int("RAG_TIMEOUT", 180)

    # --- Chunking ----------------------------------------------------------
    # Sizes are in characters. ~1200 chars is roughly 300 tokens.
    chunk_size: int = _env_int("RAG_CHUNK_SIZE", 1200)
    chunk_overlap: int = _env_int("RAG_CHUNK_OVERLAP", 180)

    # --- Retrieval ---------------------------------------------------------
    top_k: int = _env_int("RAG_TOP_K", 5)
    # Chunks scoring below this cosine similarity are discarded.
    # Raise it if the model answers from irrelevant passages.
    min_score: float = _env_float("RAG_MIN_SCORE", 0.25)
    # Small boost for chunks that literally contain the question's keywords.
    keyword_boost: float = _env_float("RAG_KEYWORD_BOOST", 0.05)

    def __post_init__(self) -> None:
        self.docs_dir = Path(self.docs_dir)
        self.index_dir = Path(self.index_dir)


CONFIG = Config()
