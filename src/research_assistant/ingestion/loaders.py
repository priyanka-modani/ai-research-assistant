from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docx import Document
from pypdf import PdfReader

from research_assistant.schemas import SourceDocument

if TYPE_CHECKING:
    from research_assistant.config import Settings


RICH_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}


def _read_local(path: Path) -> list[tuple[str, int | None]]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return [(path.read_text(encoding="utf-8"), None)]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [(json.dumps(payload, indent=2, ensure_ascii=False), None)]
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return [("", None)]
        header = " | ".join(rows[0].keys())
        separator = " | ".join("---" for _ in rows[0])
        body = [" | ".join(str(row[key]) for key in row) for row in rows]
        return [("\n".join([header, separator, *body]), None)]
    if suffix == ".docx":
        document = Document(path)
        return [("\n".join(p.text for p in document.paragraphs if p.text.strip()), None)]
    if suffix == ".pdf":
        reader = PdfReader(path)
        return [(page.extract_text() or "", number) for number, page in enumerate(reader.pages, 1)]
    raise ValueError(f"Unsupported file type: {path}")


def _read_llamaparse(path: Path, settings: Settings) -> list[tuple[str, int | None]]:
    from llama_cloud import LlamaCloud

    client = LlamaCloud(api_key=settings.llama_cloud_api_key)
    uploaded = client.files.create(file=str(path), purpose="parse")
    result = client.parsing.parse(
        file_id=uploaded.id,
        tier=settings.llamaparse_tier,
        version="latest",
        expand=["markdown"],
        output_options={"markdown": {"tables": {"output_tables_as_markdown": True}}},
    )
    pages = getattr(getattr(result, "markdown", None), "pages", []) or []
    return [(page.markdown, index) for index, page in enumerate(pages, 1)]


def load_manifest(corpus_dir: Path) -> list[SourceDocument]:
    manifest_path = corpus_dir / "manifest.json"
    manifest: list[dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [load_manifest_entry(corpus_dir, entry) for entry in manifest]


def load_manifest_entry(corpus_dir: Path, entry: dict[str, Any]) -> SourceDocument:
    path = corpus_dir / entry["source_path"]
    pages = _read_local(path)
    text = "\n\n".join(
        f"[Page {page}]\n{content}" if page is not None else content for content, page in pages
    )
    payload = {**entry, "source_path": str(path), "text": text}
    return SourceDocument.model_validate(payload)


def load_documents(corpus_dir: Path, settings: Settings) -> list[SourceDocument]:
    manifest: list[dict[str, Any]] = json.loads(
        (corpus_dir / "manifest.json").read_text(encoding="utf-8")
    )
    output: list[SourceDocument] = []
    for entry in manifest:
        path = corpus_dir / entry["source_path"]
        use_cloud = (
            settings.use_llamaparse
            and bool(settings.llama_cloud_api_key)
            and path.suffix.lower() in RICH_EXTENSIONS
        )
        pages = _read_llamaparse(path, settings) if use_cloud else _read_local(path)
        for page_text, page in pages:
            if not page_text.strip():
                continue
            payload = {**entry, "source_path": str(path), "text": page_text, "page": page}
            output.append(SourceDocument.model_validate(payload))
    return output
