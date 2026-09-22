import pytest

from research_assistant.config import Settings
from research_assistant.graph.workflow import (
    _has_explicit_negative_evidence,
    build_rag_graph,
)
from research_assistant.schemas import ChunkMetadata, RetrievedChunk


def make_chunk(text: str, study_type: str = "current_project") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"source:v1:body:1:{study_type}",
        text=text,
        metadata=ChunkMetadata(
            project_id="connecttel_cx_2026",
            document_id="source",
            document_title="Scope Status",
            artifact_type="client_email",
            version=1,
            study_type=study_type,
            source_path="current_project/scope_status.json",
            content_hash="abc",
        ),
    )


class FakeIndex:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def search(self, query: str) -> list[RetrievedChunk]:
        return self.chunks


class BypassExpectedProvider:
    def generate_json(self, system: str, user: str) -> dict:
        if "evidence sufficiency grader" in system:
            raise AssertionError("Explicit negative evidence should bypass the LLM grader")
        return {
            "answer": "No. Spain is not currently part of the approved project scope. [C1]",
            "citation_ids": ["C1"],
            "missing_evidence": [],
        }


def test_explicit_not_approved_evidence_produces_answer_instead_of_refusal():
    index = FakeIndex(
        [
            make_chunk(
                "Spain was requested for assessment. The request is not approved, and Spain "
                "has not been added to the project scope."
            )
        ]
    )
    graph = build_rag_graph(Settings(min_evidence_count=1), BypassExpectedProvider(), index)

    state = graph.invoke({"question": "Is Spain part of the approved project scope?"})

    assert state["refused"] is False
    assert state["answer"].startswith("No.")
    assert state["used_citation_ids"] == ["C1"]


@pytest.mark.parametrize(
    ("question", "statement"),
    [
        (
            "Is Chicago part of the approved project scope?",
            "Chicago has not been added to the approved project scope.",
        ),
        (
            "Has Germany been approved?",
            "Germany has not yet been approved.",
        ),
        (
            "Is the questionnaire approved?",
            "The questionnaire is not formally approved.",
        ),
        (
            "Is Spain part of the approved project scope?",
            "Spain is not part of the approved project scope.",
        ),
        (
            "Does the current project scope include Spain?",
            "The approved project scope does not include Spain.",
        ),
    ],
)
def test_rule_generalizes_to_other_subjects(question: str, statement: str):
    assert _has_explicit_negative_evidence(question, [make_chunk(statement)]) is True


@pytest.mark.parametrize(
    ("question", "statement", "study_type"),
    [
        (
            "Is France part of the approved project scope?",
            "France is approved, while Spain is not approved.",
            "current_project",
        ),
        (
            "Is Spain part of the approved project scope?",
            "Spain was requested as an additional market.",
            "current_project",
        ),
        (
            "Is Spain part of the approved project scope?",
            "The approved markets are Germany, France, and the United Kingdom.",
            "current_project",
        ),
        (
            "Is Spain part of the approved project scope?",
            "A previous study stated that Spain was not included.",
            "historical",
        ),
        (
            "What did the client request?",
            "Spain has not been approved.",
            "current_project",
        ),
    ],
)
def test_rule_does_not_trigger_without_a_direct_current_scope_statement(
    question: str, statement: str, study_type: str
):
    assert _has_explicit_negative_evidence(question, [make_chunk(statement, study_type)]) is False
