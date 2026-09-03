"""Phase 3b - persist vectors and search them.

A plain numpy matrix beats a vector database until roughly 100k chunks.
A brute-force dot product over 50k x 1024 floats takes a few milliseconds,
and there is no server to run, no schema to migrate, no version to pin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chunker import Chunk

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: dict) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors have different lengths")
        self.chunks = chunks
        self.vectors = vectors
        self.meta = meta

    # --- Persistence -------------------------------------------------------

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / VECTORS_FILE, self.vectors)
        (index_dir / CHUNKS_FILE).write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        (index_dir / META_FILE).write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        if not (index_dir / VECTORS_FILE).exists():
            raise FileNotFoundError(
                f"No index found in {index_dir}. Run 'ingest' first."
            )
        vectors = np.load(index_dir / VECTORS_FILE)
        raw_chunks = json.loads((index_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
        meta = json.loads((index_dir / META_FILE).read_text(encoding="utf-8"))
        return cls([Chunk(**c) for c in raw_chunks], vectors, meta)

    # --- Search ------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        """Cosine similarity search. Vectors are already normalized, so the
        dot product is the cosine."""
        scores = self.vectors @ query_vector
        top_k = min(top_k, len(scores))
        # argpartition is O(n) instead of sorting the whole array
        candidates = np.argpartition(-scores, top_k - 1)[:top_k]
        ordered = candidates[np.argsort(-scores[candidates])]
        return [SearchHit(self.chunks[i], float(scores[i])) for i in ordered]

    # --- Introspection -----------------------------------------------------

    def describe(self) -> str:
        sources = sorted({c.source for c in self.chunks})
        lines = [
            f"Chunks     : {len(self.chunks)}",
            f"Dimension  : {self.vectors.shape[1]}",
            f"Embed model: {self.meta.get('embed_model', 'unknown')}",
            f"Built at   : {self.meta.get('built_at', 'unknown')}",
            f"Documents  : {len(sources)}",
        ]
        lines += [f"  - {name}" for name in sources[:20]]
        if len(sources) > 20:
            lines.append(f"  ... and {len(sources) - 20} more")
        return "\n".join(lines)
