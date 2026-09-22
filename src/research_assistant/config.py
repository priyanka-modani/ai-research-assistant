from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_provider: Literal["nebius", "openai"] = "nebius"

    nebius_api_key: str = ""
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1"
    nebius_chat_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    nebius_embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    nebius_rerank_model: str = "Qwen/Qwen3-Reranker-8B"
    enable_rerank: bool = False

    openai_api_key: str = ""
    openai_chat_model: str = ""
    openai_embedding_model: str = ""

    llama_cloud_api_key: str = ""
    use_llamaparse: bool = True
    llamaparse_tier: Literal["fast", "cost_effective", "agentic", "agentic_plus"] = "agentic"
    mem0_api_key: str = ""

    chroma_path: Path = Path("storage/chroma")
    chroma_collection: str = "connecttel_research_assistant"
    chunks_path: Path = Path("storage/chunks.json")
    top_k_dense: int = Field(8, ge=1, le=50)
    top_k_sparse: int = Field(8, ge=1, le=50)
    top_k_final: int = Field(5, ge=1, le=20)
    min_evidence_count: int = Field(1, ge=1, le=10)

    @property
    def active_chat_model(self) -> str:
        return self.nebius_chat_model if self.model_provider == "nebius" else self.openai_chat_model

    @property
    def active_embedding_model(self) -> str:
        return (
            self.nebius_embedding_model
            if self.model_provider == "nebius"
            else self.openai_embedding_model
        )

    def validate_for_runtime(self) -> None:
        key = self.nebius_api_key if self.model_provider == "nebius" else self.openai_api_key
        if not key:
            raise ValueError(f"Missing API key for MODEL_PROVIDER={self.model_provider}")
        if not self.active_chat_model or not self.active_embedding_model:
            raise ValueError("Chat and embedding model names must be configured")


@lru_cache
def get_settings() -> Settings:
    return Settings()
