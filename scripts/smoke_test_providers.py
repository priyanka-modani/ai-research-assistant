import os
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from llama_cloud import LlamaCloud
from openai import OpenAI


def test_nebius() -> None:
    client = OpenAI(
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
    )
    embedding = client.embeddings.create(
        model=os.environ["NEBIUS_EMBEDDING_MODEL"], input=["ConnectTel smoke test"]
    )
    print(f"Nebius embeddings: OK ({len(embedding.data[0].embedding)} dimensions)")
    response = client.chat.completions.create(
        model=os.environ["NEBIUS_CHAT_MODEL"],
        temperature=0,
        messages=[{"role": "user", "content": "Reply with only: OK"}],
    )
    print(f"Nebius chat: {response.choices[0].message.content}")


def test_nebius_rerank() -> None:
    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")
    response = requests.post(
        f"{base_url}/rerank",
        headers={
            "Authorization": f"Bearer {os.environ['NEBIUS_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": os.getenv("NEBIUS_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B"),
            "query": "Which market was requested?",
            "documents": ["Spain was requested.", "Germany is in the baseline."],
        },
        timeout=60,
    )
    if response.ok:
        print("Nebius rerank: OK")
        return
    print(
        f"Nebius rerank: unavailable ({response.status_code}). "
        "Keep ENABLE_RERANK=false. Response: " + response.text[:300]
    )


def test_llamaparse() -> None:
    client = LlamaCloud(api_key=os.environ["LLAMA_CLOUD_API_KEY"])
    with tempfile.TemporaryDirectory() as directory:
        test_file = Path(directory) / "smoke_test.md"
        test_file.write_text("# ConnectTel\n\nLlamaParse credential smoke test.", encoding="utf-8")
        uploaded = client.files.create(file=str(test_file), purpose="parse")
        result = client.parsing.parse(
            file_id=uploaded.id, tier="fast", version="latest", expand=["text"]
        )
        print(f"LlamaParse: OK (job result type {type(result).__name__})")


if __name__ == "__main__":
    load_dotenv()
    test_nebius()
    test_nebius_rerank()
    test_llamaparse()
