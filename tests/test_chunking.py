from research_assistant.ingestion.chunking import chunk_document
from research_assistant.schemas import SourceDocument


def test_chunk_ids_are_stable_and_metadata_survives():
    document = SourceDocument(
        project_id="p1",
        document_id="brief",
        document_title="Brief",
        artifact_type="client_brief",
        version=2,
        study_type="current_project",
        source_path="brief.md",
        text="# Scope\n\nGermany, France, and the UK are in scope.",
        market=["Germany", "France", "UK"],
    )
    first = chunk_document(document)
    second = chunk_document(document)
    assert first[0].chunk_id == second[0].chunk_id
    assert first[0].metadata.version == 2
    assert first[0].metadata.section == "Scope"
    assert first[0].metadata.market == ["Germany", "France", "UK"]
