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
            return self._chat_ollama(messages)
        if self.config.llm_backend == "openai":
            return self._chat_openai(messages)
        raise ValueError(f"Unknown LLM backend: {self.config.llm_backend}")

    def _chat_ollama(self, messages: list[dict]) -> str:
        response = requests.post(
            self.config.llm_url,
            json={
                "model": self.config.llm_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.config.llm_temperature},
            },
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def _chat_openai(self, messages: list[dict]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"

        response = requests.post(
            self.config.llm_url,
            headers=headers,
            json={
                "model": self.config.llm_model,
                "messages": messages,
                "temperature": self.config.llm_temperature,
                "stream": False,
            },
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
