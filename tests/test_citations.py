from research_assistant.citations import make_citations, valid_citation_ids
from research_assistant.schemas import ChunkMetadata, RetrievedChunk


def test_invalid_citations_are_removed():
    chunk = RetrievedChunk(
        chunk_id="x",
        text="Spain was requested, not approved.",
        metadata=ChunkMetadata(
            project_id="p",
            document_id="email",
            document_title="Client email",
            artifact_type="email",
            version=1,
            study_type="current_project",
            source_path="email.txt",
            content_hash="abc",
        ),
    )
    citations = make_citations([chunk])
    assert valid_citation_ids(["C1", "C99"], citations) == ["C1"]
