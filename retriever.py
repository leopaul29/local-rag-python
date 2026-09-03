"""Phase 4 - find the passages that should answer the question.

This is where RAG quality is won or lost. Debug this layer in isolation
with the `search` command before you ever look at the LLM's output.
"""

from __future__ import annotations

import re

from config import Config
from embedder import Embedder
from store import SearchHit, VectorStore

_WORD = re.compile(r"\w{4,}", re.UNICODE)


def _keywords(text: str) -> set[str]:
    """Content words only. The length filter is a cheap stopword remover."""
    return {w.lower() for w in _WORD.findall(text)}


class Retriever:
    def __init__(self, config: Config, store: VectorStore, embedder: Embedder) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchHit]:
        top_k = top_k or self.config.top_k

        query_vector = self.embedder.embed_query(question)
        # Over-fetch, then re-rank and filter, so the threshold does not
        # silently leave us with fewer results than requested.
        hits = self.store.search(query_vector, top_k * 4)

        hits = self._apply_keyword_boost(question, hits)
        hits.sort(key=lambda h: h.score, reverse=True)

        kept = [h for h in hits if h.score >= self.config.min_score]
        return kept[:top_k]

    def _apply_keyword_boost(
        self, question: str, hits: list[SearchHit]
    ) -> list[SearchHit]:
        """Nudge up chunks that literally contain the question's terms.

        Pure vector search is weak on rare tokens: proper nouns, product
        references, invoice numbers. This is a poor man's hybrid search.
        """
        boost = self.config.keyword_boost
        if boost <= 0:
            return hits

        question_words = _keywords(question)
        if not question_words:
            return hits

        for hit in hits:
            overlap = question_words & _keywords(hit.chunk.text)
            hit.score += boost * (len(overlap) / len(question_words))
        return hits


def format_context(hits: list[SearchHit]) -> str:
    """Render retrieved chunks into the block injected in the prompt."""
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(f"[{index}] Source: {hit.chunk.label()}\n{hit.chunk.text}")
    return "\n\n---\n\n".join(blocks)
