from research_assistant.schemas import Citation, RetrievedChunk


def make_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for index, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        location = f"page {meta.page}" if meta.page else meta.section
        citations.append(
            Citation(
                citation_id=f"C{index}",
                chunk_id=chunk.chunk_id,
                document_title=meta.document_title,
                document_id=meta.document_id,
                version=meta.version,
                page_or_section=location,
                study_type=meta.study_type,
                supporting_excerpt=chunk.text[:500],
            )
        )
    return citations


def valid_citation_ids(requested: list[str], citations: list[Citation]) -> list[str]:
    allowed = {citation.citation_id for citation in citations}
    return [citation_id for citation_id in requested if citation_id in allowed]
