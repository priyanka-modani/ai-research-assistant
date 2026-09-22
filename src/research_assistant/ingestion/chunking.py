import hashlib
import re

from research_assistant.schemas import ChunkMetadata, DocumentChunk, SourceDocument

HEADING_RE = re.compile(r"(?m)^(#{1,4}\s+.+|[A-Z][A-Z0-9 /&()-]{5,})$")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(text: str) -> list[tuple[str, str]]:
    cleaned = clean_text(text)
    matches = list(HEADING_RE.finditer(cleaned))
    if not matches:
        return [("Document", cleaned)]
    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        sections.append(("Preamble", cleaned[: matches[0].start()].strip()))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        title = match.group(0).lstrip("#").strip()
        body = cleaned[start:end].strip()
        sections.append((title, body))
    return [(title, body) for title, body in sections if body]


def _windows(words: list[str], max_words: int, overlap_words: int) -> list[list[str]]:
    if len(words) <= max_words:
        return [words]
    step = max_words - overlap_words
    return [words[start : start + max_words] for start in range(0, len(words), step)]


def chunk_document(
    document: SourceDocument, max_words: int = 360, overlap_words: int = 55
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for section, body in split_sections(document.text):
        words = body.split()
        for ordinal, window in enumerate(_windows(words, max_words, overlap_words), 1):
            text = " ".join(window).strip()
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk_id = (
                f"{document.document_id}:v{document.version}:{section}:{ordinal}:{digest[:10]}"
            )
            metadata = ChunkMetadata(
                project_id=document.project_id,
                document_id=document.document_id,
                document_title=document.document_title,
                artifact_type=document.artifact_type,
                version=document.version,
                study_type=document.study_type,
                section=section,
                page=document.page,
                market=document.market,
                topic=document.topic,
                date=document.date,
                source_path=document.source_path,
                content_hash=digest,
            )
            chunks.append(DocumentChunk(chunk_id=chunk_id, text=text, metadata=metadata))
    return chunks


def chunk_documents(documents: list[SourceDocument]) -> list[DocumentChunk]:
    return [chunk for document in documents for chunk in chunk_document(document)]
