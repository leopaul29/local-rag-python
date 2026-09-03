#!/usr/bin/env python3
"""Command line entry point for the local RAG.

Usage:
    python main.py ingest --docs ./documents
    python main.py search "your question"     # retrieval only, for debugging
    python main.py ask "your question"
    python main.py chat
    python main.py info
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from chunker import chunk_documents
from config import CONFIG, Config
from embedder import Embedder
from generator import Generator
from loaders import load_directory
from retriever import Retriever
from store import VectorStore


# --- Commands --------------------------------------------------------------


def cmd_ingest(config: Config, args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs) if args.docs else config.docs_dir
    print(f"Reading documents from {docs_dir.resolve()}")

    documents = list(load_directory(docs_dir))
    if not documents:
        print("No readable document found. Supported: .pdf .docx .txt .md")
        return 1
    print(f"Loaded {len(documents)} document units")

    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    print(f"Produced {len(chunks)} chunks")
    if args.preview:
        for chunk in chunks[:3]:
            print(f"\n--- {chunk.label()} ---\n{chunk.text[:400]}...")
        print()

    print(f"Embedding with '{config.embed_model}' ({config.embed_backend})")
    embedder = Embedder(config)
    vectors = embedder.embed_documents([c.text for c in chunks])

    meta = {
        "embed_model": config.embed_model,
        "embed_backend": config.embed_backend,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    VectorStore(chunks, vectors, meta).save(config.index_dir)
    print(f"Index saved to {config.index_dir.resolve()}")
    return 0


def cmd_search(config: Config, args: argparse.Namespace) -> int:
    """Retrieval only. If the answer is not in these passages, the LLM
    cannot possibly produce it. Always debug here first."""
    retriever = _build_retriever(config)
    hits = retriever.retrieve(args.question, top_k=args.top_k)

    if not hits:
        print("No chunk passed the relevance threshold. Try lowering RAG_MIN_SCORE.")
        return 1

    for index, hit in enumerate(hits, start=1):
        print(f"\n[{index}] {hit.chunk.label()}  (score {hit.score:.3f})")
        print(hit.chunk.text[:600])
    return 0


def cmd_ask(config: Config, args: argparse.Namespace) -> int:
    retriever = _build_retriever(config)
    generator = Generator(config)

    hits = retriever.retrieve(args.question)
    answer = generator.answer(args.question, hits)

    print(f"\n{answer}\n")
    if hits:
        print("Sources:")
        for index, hit in enumerate(hits, start=1):
            print(f"  [{index}] {hit.chunk.label()}  (score {hit.score:.3f})")
    return 0


def cmd_chat(config: Config, args: argparse.Namespace) -> int:
    """Interactive loop. Each question is retrieved independently, so
    follow-up questions must be self-contained."""
    retriever = _build_retriever(config)
    generator = Generator(config)
    print("Ask a question, or type 'exit' to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if not question:
            continue

        hits = retriever.retrieve(question)
        print(f"\n{generator.answer(question, hits)}\n")
        for index, hit in enumerate(hits, start=1):
            print(f"  [{index}] {hit.chunk.label()}  ({hit.score:.3f})")
        print()


def cmd_info(config: Config, args: argparse.Namespace) -> int:
    store = VectorStore.load(config.index_dir)
    print(store.describe())
    print(f"\nLLM        : {config.llm_model} ({config.llm_backend})")
    print(f"LLM URL    : {config.llm_url}")
    print(f"top_k      : {config.top_k}   min_score: {config.min_score}")
    return 0


# --- Helpers ---------------------------------------------------------------


def _build_retriever(config: Config) -> Retriever:
    store = VectorStore.load(config.index_dir)
    indexed_model = store.meta.get("embed_model")
    if indexed_model and indexed_model != config.embed_model:
        # Vectors from two different models are not comparable at all
        print(
            f"WARNING: index was built with '{indexed_model}' but config uses "
            f"'{config.embed_model}'. Re-run ingest.",
            file=sys.stderr,
        )
    return Retriever(config, store, Embedder(config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local RAG over personal documents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Build the index from documents")
    ingest.add_argument("--docs", help="Documents directory")
    ingest.add_argument(
        "--preview", action="store_true", help="Print the first chunks produced"
    )
    ingest.set_defaults(func=cmd_ingest)

    search = subparsers.add_parser("search", help="Show retrieved passages only")
    search.add_argument("question")
    search.add_argument("--top-k", type=int, default=None)
    search.set_defaults(func=cmd_search)

    ask = subparsers.add_parser("ask", help="Ask a single question")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    chat = subparsers.add_parser("chat", help="Interactive question loop")
    chat.set_defaults(func=cmd_chat)

    info = subparsers.add_parser("info", help="Describe the current index")
    info.set_defaults(func=cmd_info)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(CONFIG, args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
