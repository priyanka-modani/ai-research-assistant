import json
import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from research_assistant.citations import make_citations, valid_citation_ids
from research_assistant.config import Settings
from research_assistant.prompts.answer_generation import ANSWER_GENERATION_SYSTEM_PROMPT
from research_assistant.prompts.evidence_grading import EVIDENCE_GRADING_SYSTEM_PROMPT
from research_assistant.providers import ModelProvider
from research_assistant.retrieval.index import HybridIndex
from research_assistant.schemas import Citation, GroundedAnswer, RetrievedChunk

APPROVAL_QUERY_TERMS = (
    "approved",
    "approval",
    "scope",
    "included",
    "inclusion",
    "part of",
)
SUBJECT_STOPWORDS = {
    "a",
    "an",
    "are",
    "approved",
    "approval",
    "be",
    "been",
    "current",
    "currently",
    "did",
    "do",
    "does",
    "has",
    "have",
    "in",
    "include",
    "included",
    "inclusion",
    "is",
    "of",
    "part",
    "project",
    "scope",
    "study",
    "the",
    "was",
    "were",
}


class RAGState(TypedDict, total=False):
    question: str
    query_type: str
    evidence: list[RetrievedChunk]
    citations: list[Citation]
    evidence_sufficient: bool
    missing_evidence: list[str]
    answer: str
    used_citation_ids: list[str]
    refused: bool
    errors: list[str]


def _classify(question: str) -> str:
    lowered = question.lower()
    if any(word in lowered for word in ("historical", "previous", "prior", "reuse")):
        return "historical"
    if any(word in lowered for word in ("risk", "conflict", "approved", "scope", "delay", "schedule", "timeline", "deadline", "task sequence",)):
        return "risk"
    if any(word in lowered for word in ("compare", "difference", "across")):
        return "comparison"
    return "current_project"


def _format_evidence(chunks: list[RetrievedChunk], citations: list[Citation]) -> str:
    citation_by_chunk = {citation.chunk_id: citation for citation in citations}
    blocks = []
    for chunk in chunks:
        citation = citation_by_chunk[chunk.chunk_id]
        blocks.append(
            "\n".join(
                [
                    f"[{citation.citation_id}] {citation.document_title}",
                    f"Document ID: {citation.document_id}; version: {citation.version}",
                    f"Study type: {citation.study_type}; location: {citation.page_or_section}",
                    chunk.text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _question_subject(question: str) -> str | None:
    words = re.findall(r"[A-Za-z0-9_-]+", question.lower())
    subject_words = [word for word in words if word not in SUBJECT_STOPWORDS]
    return " ".join(subject_words) or None


def _subject_has_negative_status(sentence: str, subject: str) -> bool:
    subject_pattern = r"\b" + r"\s+".join(re.escape(word) for word in subject.split()) + r"\b"
    subject_detail = (
        r"(?:'s)?(?:\s+(?:addition|approval|inclusion|market|participation|request|scope)){0,3}"
    )
    optional_adverb = r"(?:currently\s+|formally\s+|still\s+)*"
    post_negative_adverb = r"(?:currently\s+|formally\s+|still\s+|yet\s+)*"
    scope_label = r"(?:the\s+)?(?:(?:approved|current)\s+)?(?:project\s+)?scope"
    negative_status = (
        rf"(?:approved|accepted|added|authorized|confirmed|included|in\s+{scope_label}|"
        rf"part\s+of\s+{scope_label})"
    )
    patterns = (
        rf"{subject_pattern}{subject_detail}\s+(?:is|are|was|were)\s+"
        rf"{optional_adverb}(?:not|never)\s+{post_negative_adverb}{negative_status}",
        rf"{subject_pattern}{subject_detail}\s+(?:has|have|had)\s+(?:not|never)\s+"
        rf"(?:yet\s+)?(?:been\s+)?{negative_status}",
        rf"{subject_pattern}{subject_detail}\s+remains?\s+"
        rf"(?:outside\s+{scope_label}|excluded)",
        rf"{subject_pattern}{subject_detail}\s+(?:is|are|was|were)\s+"
        rf"(?:currently\s+|still\s+)?(?:excluded|outside\s+{scope_label}|pending)",
        rf"{scope_label}\s+(?:does|do|did)\s+(?:not|never)\s+include\s+{subject_pattern}",
    )
    return any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in patterns)


def _has_explicit_negative_evidence(question: str, evidence: list[RetrievedChunk]) -> bool:
    lowered_question = question.lower()
    if not any(term in lowered_question for term in APPROVAL_QUERY_TERMS):
        return False

    subject = _question_subject(question)
    if not subject:
        return False

    for chunk in evidence:
        if chunk.metadata.study_type != "current_project":
            continue
        sentences = re.split(r"(?<=[.!?;])\s+|\n+", chunk.text)
        if any(_subject_has_negative_status(sentence, subject) for sentence in sentences):
            return True
    return False


def build_rag_graph(settings: Settings, provider: ModelProvider, index: HybridIndex):
    def classify_query(state: RAGState) -> dict[str, Any]:
        return {"query_type": _classify(state["question"]), "errors": []}

    def retrieve(state: RAGState) -> dict[str, Any]:
        evidence = index.search(state["question"])
        return {"evidence": evidence, "citations": make_citations(evidence)}

    def grade_evidence(state: RAGState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        if len(evidence) < settings.min_evidence_count:
            return {
                "evidence_sufficient": False,
                "missing_evidence": ["No sufficiently relevant source passage was retrieved."],
            }
        if _has_explicit_negative_evidence(state["question"], evidence):
            return {
                "evidence_sufficient": True,
                "missing_evidence": [],
            }
        citations = state["citations"]
        evidence_text = _format_evidence(evidence, citations)
        payload = provider.generate_json(
            system=EVIDENCE_GRADING_SYSTEM_PROMPT,
            user=f"Question: {state['question']}\n\nEvidence:\n{evidence_text}",
        )
        return {
            "evidence_sufficient": bool(payload.get("sufficient", False)),
            "missing_evidence": list(payload.get("missing_evidence", [])),
        }

    def generate_answer(state: RAGState) -> dict[str, Any]:
        evidence = state["evidence"]
        citations = state["citations"]
        payload = provider.generate_json(
            system=ANSWER_GENERATION_SYSTEM_PROMPT,
            user=(
                f"Question: {state['question']}\n\n"
                f"Evidence:\n{_format_evidence(evidence, citations)}"
            ),
        )
        parsed = GroundedAnswer.model_validate(payload)
        ids = valid_citation_ids(parsed.citation_ids, citations)
        if not ids:
            return {
                "answer": "I could not produce an answer with valid supporting citations.",
                "used_citation_ids": [],
                "missing_evidence": ["The generated answer did not include a valid citation."],
                "refused": True,
            }
        return {
            "answer": parsed.answer,
            "used_citation_ids": ids,
            "missing_evidence": parsed.missing_evidence,
            "refused": False,
        }

    def refuse(state: RAGState) -> dict[str, Any]:
        return {
            "answer": (
                "I could not find enough information in the available project documents "
                "to answer this question."
            ),
            "used_citation_ids": [],
            "refused": True,
        }

    def route_after_grade(state: RAGState) -> Literal["generate_answer", "refuse"]:
        return "generate_answer" if state.get("evidence_sufficient") else "refuse"

    builder = StateGraph(RAGState)
    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_evidence", grade_evidence)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("refuse", refuse)
    builder.add_edge(START, "classify_query")
    builder.add_edge("classify_query", "retrieve")
    builder.add_edge("retrieve", "grade_evidence")
    builder.add_conditional_edges("grade_evidence", route_after_grade)
    builder.add_edge("generate_answer", END)
    builder.add_edge("refuse", END)
    return builder.compile()


def serialize_state(state: RAGState) -> dict[str, Any]:
    result = dict(state)
    result["evidence"] = [item.model_dump() for item in state.get("evidence", [])]
    result["citations"] = [item.model_dump() for item in state.get("citations", [])]
    return json.loads(json.dumps(result, default=str))
