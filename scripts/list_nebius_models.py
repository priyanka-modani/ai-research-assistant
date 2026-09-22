"""List the models actually visible to the configured Nebius account."""

import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()
    client = OpenAI(
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
    )
    models = sorted(model.id for model in client.models.list().data)
    print("Models visible to this API key:")
    for model in models:
        lowered = model.lower()
        if "rerank" in lowered:
            category = "RERANK"
        elif "embed" in lowered or "bge" in lowered or "e5-" in lowered:
            category = "EMBEDDING"
        else:
            category = "TEXT/VISION - confirm in dashboard"
        print(f"[{category}] {model}")
    print("\nDocumented reranker endpoint model: Qwen/Qwen3-Reranker-8B")


if __name__ == "__main__":
    main()
