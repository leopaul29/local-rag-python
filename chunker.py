"""Phase 2 - split documents into retrievable chunks.

Strategy: never cut in the middle of a paragraph if it can be avoided.
Paragraphs are accumulated until the size limit is reached; only oversized
paragraphs are split further, on sentence boundaries.

Japanese-aware: Japanese sentences end with 。！？ and are NOT followed by
whitespace, and Japanese has no spaces between words. Both the sentence
splitter and the overlap logic handle that explicitly, while still working
on Latin-script text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable

from loaders import RawDocument

# Japanese sentence terminators. Note that 、 is a comma, not a terminator.
JA_SENTENCE_END = "。．！？"
# Closing marks that belong to the sentence they follow: 「これだ。」
CLOSING_MARKS = "」』）】〉》＞”’\"')"
# Latin terminators only end a sentence when whitespace follows, so that
# "3.5" or "M. Dupont" is not treated as a sentence break.
LATIN_SENTENCE_END = ".!?;:"

# Chunks shorter than this are dropped as noise. Japanese is information
# dense, so the floor is low on purpose.
MIN_CHUNK_CHARS = 20


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


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, Japanese and Latin alike.

    Each returned sentence keeps its own trailing whitespace, so the caller
    can concatenate sentences back together without inserting spaces that
    do not belong in Japanese.
    """
    sentences: list[str] = []
    start = 0
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        # Japanese: terminator alone is enough, no whitespace required
        if char in JA_SENTENCE_END:
            end = index + 1
            # Keep trailing quotes and brackets attached: 「そうだ。」
            while end < length and text[end] in CLOSING_MARKS:
                end += 1
            while end < length and text[end].isspace():
                end += 1
            sentences.append(text[start:end])
            start = index = end
            continue

        # Latin: terminator must be followed by whitespace
        if char in LATIN_SENTENCE_END and index + 1 < length and text[index + 1].isspace():
            end = index + 1
            while end < length and text[end].isspace():
                end += 1
            sentences.append(text[start:end])
            start = index = end
            continue

        index += 1

    if start < length:
        sentences.append(text[start:])

    return [s for s in sentences if s.strip()]


def _split_oversized(paragraph: str, chunk_size: int) -> list[str]:
    """Break a paragraph longer than chunk_size into groups of sentences."""
    pieces: list[str] = []
    buffer = ""

    for sentence in split_sentences(paragraph):
        # A single sentence longer than the limit gets a hard character cut
        if len(sentence) > chunk_size:
            if buffer.strip():
                pieces.append(buffer.strip())
                buffer = ""
            for start in range(0, len(sentence), chunk_size):
                pieces.append(sentence[start : start + chunk_size].strip())
            continue

        if buffer and len(buffer) + len(sentence) > chunk_size:
            pieces.append(buffer.strip())
            buffer = sentence
        else:
            # No separator added: sentences already carry their own spacing
            buffer += sentence

    if buffer.strip():
        pieces.append(buffer.strip())
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Take the last `overlap` characters to prepend to the next chunk.

    Preference order: restart at a sentence boundary, then at a word
    boundary, then anywhere. Japanese has no spaces, so the raw cut is the
    normal outcome there and is harmless.
    """
    if overlap <= 0 or len(text) <= overlap:
        return text

    tail = text[-overlap:]

    for index, char in enumerate(tail[:-1]):
        if char in JA_SENTENCE_END:
            candidate = tail[index + 1 :].lstrip("".join(CLOSING_MARKS)).lstrip()
            if candidate:
                return candidate

    space = tail.find(" ")
    if space != -1:
        return tail[space + 1 :]

    return tail


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

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


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
