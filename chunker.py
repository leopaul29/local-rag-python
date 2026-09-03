"""Phase 2 - split documents into retrievable chunks.

Strategy: never cut in the middle of a paragraph if it can be avoided.
Paragraphs are accumulated until the size limit is reached; only oversized
paragraphs are split further, on sentence boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable

from loaders import RawDocument

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


@dataclass
class Chunk:
    id: int
    source: str
    page: int | None
    text: str

    def label(self) -> str:
        """Short human-readable origin, used in the prompt and in citations."""
        return f"{self.source} p.{self.page}" if self.page else self.source

    def to_dict(self) -> dict:
        return asdict(self)


def _split_oversized(paragraph: str, chunk_size: int) -> list[str]:
    """Break a paragraph that is longer than chunk_size into sentence groups."""
    sentences = _SENTENCE_END.split(paragraph)
    pieces: list[str] = []
    buffer = ""

    for sentence in sentences:
        # A single sentence longer than the limit gets a hard character cut
        if len(sentence) > chunk_size:
            if buffer:
                pieces.append(buffer.strip())
                buffer = ""
            for start in range(0, len(sentence), chunk_size):
                pieces.append(sentence[start : start + chunk_size].strip())
            continue

        if len(buffer) + len(sentence) + 1 > chunk_size:
            pieces.append(buffer.strip())
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()

    if buffer.strip():
        pieces.append(buffer.strip())
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Take the last `overlap` characters, snapped to a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Turn one document's text into overlapping chunks."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > chunk_size:
            units.extend(_split_oversized(paragraph, chunk_size))
        else:
            units.append(paragraph)

    chunks: list[str] = []
    buffer = ""

    for unit in units:
        if buffer and len(buffer) + len(unit) + 2 > chunk_size:
            chunks.append(buffer.strip())
            # Carry the tail of the previous chunk so context is not lost
            # at the seam between two chunks.
            buffer = f"{_tail(buffer, overlap)}\n\n{unit}"
        else:
            buffer = f"{buffer}\n\n{unit}".strip()

    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if len(c) > 40]  # drop near-empty fragments


def chunk_documents(
    documents: Iterable[RawDocument], chunk_size: int, overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for piece in split_text(document.text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    id=len(chunks),
                    source=document.source,
                    page=document.page,
                    text=piece,
                )
            )
    return chunks
