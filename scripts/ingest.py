import json
from pathlib import Path

import chromadb

from research_assistant.config import get_settings
from research_assistant.ingestion.chunking import chunk_documents
from research_assistant.ingestion.loaders import load_documents
from research_assistant.providers import build_provider


def main() -> None:
    settings = get_settings()
    provider = build_provider(settings)
    documents = load_documents(Path("data/corpus"), settings)
    chunks = chunk_documents(documents)
    if not chunks:
        raise RuntimeError("No chunks produced from the corpus")

    settings.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    settings.chunks_path.write_text(
        json.dumps([chunk.model_dump() for chunk in chunks], indent=2), encoding="utf-8"
    )

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass
    collection = client.create_collection(
        settings.chroma_collection, metadata={"hnsw:space": "cosine"}
    )

    batch_size = 32
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = provider.embed([chunk.text for chunk in batch])
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": chunk.metadata.document_id,
                    "document_title": chunk.metadata.document_title,
                    "version": chunk.metadata.version,
                    "study_type": chunk.metadata.study_type,
                    "section": chunk.metadata.section or "",
                }
                for chunk in batch
            ],
        )
    print(f"Indexed {len(chunks)} chunks from {len(documents)} document/page records")


if __name__ == "__main__":
    main()
