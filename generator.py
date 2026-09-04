"""Phase 5 - build a grounded prompt and call the local LLM.

This is the part you already had working. The only real addition is a
system prompt that forbids answering from the model's own knowledge.
"""

from __future__ import annotations

import requests

from config import Config
from retriever import format_context
from store import SearchHit

SYSTEM_PROMPT = """You are a document question-answering assistant.

Rules:
- Answer ONLY from the context passages provided by the user.
- If the context does not contain the answer, reply exactly: "Not found in the documents."
- Never use outside knowledge and never guess.
- Cite the passages you used with their bracket numbers, e.g. [1], [2].
- Answer in the same language as the question.
- Be concise and factual."""

USER_TEMPLATE = """Context passages:

{context}

---

Question: {question}"""


def _extract_content(data: dict) -> str:
    """Pull the answer text out of whatever shape the server replied with.

    Ollama returns {"message": {"content": ...}}, OpenAI-compatible servers
    return {"choices": [{"message": {"content": ...}}]}. Accepting both means
    a wrong RAG_LLM_BACKEND value cannot break the run, and an unknown shape
    reports what it actually received instead of raising a bare KeyError.
    """
    if isinstance(data, dict):
        message = data.get("message")
        if isinstance(message, dict) and "content" in message:
            return str(message["content"]).strip()

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                if isinstance(choice.get("message"), dict):
                    return str(choice["message"].get("content", "")).strip()
                if "text" in choice:  # legacy completions endpoint
                    return str(choice["text"]).strip()

        if isinstance(data.get("response"), str):  # Ollama /api/generate
            return data["response"].strip()

    keys = list(data)[:10] if isinstance(data, dict) else type(data).__name__
    raise RuntimeError(
        "Could not find the answer in the server response.\n"
        f"Top-level keys received: {keys}\n"
        "If they include 'choices', the server is OpenAI-compatible: set "
        "RAG_LLM_BACKEND=openai and point RAG_LLM_URL at /v1/chat/completions."
    )


class Generator:
    def __init__(self, config: Config) -> None:
        self.config = config

    def answer(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return "Not found in the documents. (No passage passed the relevance threshold.)"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    context=format_context(hits), question=question
                ),
            },
        ]
        return self._chat(messages)

    def _chat(self, messages: list[dict]) -> str:
        if self.config.llm_backend == "ollama":
            payload = {
                "model": self.config.llm_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.config.llm_temperature},
            }
        elif self.config.llm_backend == "openai":
            payload = {
                "model": self.config.llm_model,
                "messages": messages,
                "temperature": self.config.llm_temperature,
                "stream": False,
            }
        else:
            raise ValueError(f"Unknown LLM backend: {self.config.llm_backend}")

        headers = {"Content-Type": "application/json"}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"

        response = requests.post(
            self.config.llm_url,
            headers=headers,
            json=payload,
            timeout=self.config.request_timeout,
        )

        # Do not use raise_for_status(): the body carries the real reason
        # ("model not found", "context length exceeded") and we want to show it.
        if response.status_code >= 400:
            raise RuntimeError(
                f"LLM server returned HTTP {response.status_code} at "
                f"{self.config.llm_url}\n{response.text[:500]}"
            )

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(
                f"LLM server did not return JSON. First 500 characters:\n"
                f"{response.text[:500]}"
            ) from None

        # Ollama can answer 200 with an error object instead of a message
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"LLM server error: {data['error']}")

        return _extract_content(data)
