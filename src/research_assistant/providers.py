import json
import warnings
from abc import ABC, abstractmethod
from typing import Any

import requests
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from research_assistant.config import Settings


class ModelProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def generate_json(self, system: str, user: str) -> dict[str, Any]: ...

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        return [(index, 0.0) for index in range(len(documents))]


class OpenAICompatibleProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        chat_model: str,
        embedding_model: str,
        rerank_model: str = "",
    ) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.rerank_warning: str | None = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def generate_json(self, system: str, user: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if not self.rerank_model:
            return super().rerank(query, documents)
        try:
            response = requests.post(
                f"{self.base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.rerank_model, "query": query, "documents": documents},
                timeout=60,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if not results:
                raise ValueError("Rerank response did not contain results")
            self.rerank_warning = None
            return [(int(item["index"]), float(item["relevance_score"])) for item in results]
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            self.rerank_warning = (
                f"Nebius reranking was unavailable ({exc}). "
                "Continuing with Reciprocal Rank Fusion results."
            )
            warnings.warn(self.rerank_warning, RuntimeWarning, stacklevel=2)
            return super().rerank(query, documents)


def build_provider(settings: Settings) -> ModelProvider:
    settings.validate_for_runtime()
    if settings.model_provider == "nebius":
        return OpenAICompatibleProvider(
            api_key=settings.nebius_api_key,
            base_url=settings.nebius_base_url,
            chat_model=settings.nebius_chat_model,
            embedding_model=settings.nebius_embedding_model,
            rerank_model=settings.nebius_rerank_model if settings.enable_rerank else "",
        )
    return OpenAICompatibleProvider(
        api_key=settings.openai_api_key,
        base_url=None,
        chat_model=settings.openai_chat_model,
        embedding_model=settings.openai_embedding_model,
    )
