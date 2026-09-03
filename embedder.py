"""Phase 3a - turn text into vectors.

Three interchangeable backends. The rest of the code only ever sees
`embed_documents` and `embed_query`, so swapping models is a config change.
"""

from __future__ import annotations

import numpy as np
import requests

from config import Config


class Embedder:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._local_model = None  # lazily loaded sentence-transformers model

    # --- Public API --------------------------------------------------------

    def embed_documents(self, texts: list[str], verbose: bool = True) -> np.ndarray:
        """Embed a corpus, in batches, and return an (n, dim) float32 matrix."""
        vectors: list[np.ndarray] = []
        batch_size = self.config.embed_batch_size
        prefixed = [self.config.passage_prefix + t for t in texts]

        for start in range(0, len(prefixed), batch_size):
            batch = prefixed[start : start + batch_size]
            vectors.append(self._embed(batch))
            if verbose:
                done = min(start + batch_size, len(prefixed))
                print(f"  embedded {done}/{len(prefixed)} chunks", end="\r")

        if verbose:
            print()
        return normalize(np.vstack(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single question and return a (dim,) unit vector."""
        vector = self._embed([self.config.query_prefix + text])
        return normalize(vector)[0]

    # --- Backends ----------------------------------------------------------

    def _embed(self, batch: list[str]) -> np.ndarray:
        backend = self.config.embed_backend
        if backend == "ollama":
            return self._embed_ollama(batch)
        if backend == "openai":
            return self._embed_openai(batch)
        if backend == "sentence-transformers":
            return self._embed_local(batch)
        raise ValueError(f"Unknown embedding backend: {backend}")

    def _embed_ollama(self, batch: list[str]) -> np.ndarray:
        response = requests.post(
            self.config.embed_url,
            json={"model": self.config.embed_model, "input": batch},
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        # /api/embed returns "embeddings", the legacy /api/embeddings returns "embedding"
        embeddings = payload.get("embeddings") or [payload["embedding"]]
        return np.asarray(embeddings, dtype=np.float32)

    def _embed_openai(self, batch: list[str]) -> np.ndarray:
        headers = {"Content-Type": "application/json"}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"

        response = requests.post(
            self.config.embed_url,
            headers=headers,
            json={"model": self.config.embed_model, "input": batch},
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        # The API may reorder results, so sort by the returned index
        data.sort(key=lambda item: item["index"])
        return np.asarray([item["embedding"] for item in data], dtype=np.float32)

    def _embed_local(self, batch: list[str]) -> np.ndarray:
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self.config.embed_model)
        return np.asarray(
            self._local_model.encode(batch, show_progress_bar=False),
            dtype=np.float32,
        )


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so that a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)
