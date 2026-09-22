from typing import Any, Literal

from pydantic import BaseModel, Field

StudyType = Literal["current_project", "historical"]


class SourceDocument(BaseModel):
    project_id: str
    document_id: str
    document_title: str
    artifact_type: str
    version: int = 1
    study_type: StudyType
    source_path: str
    text: str
    section: str | None = None
    page: int | None = None
    market: list[str] = Field(default_factory=list)
    topic: list[str] = Field(default_factory=list)
    date: str | None = None


class ChunkMetadata(BaseModel):
    project_id: str
    document_id: str
    document_title: str
    artifact_type: str
    version: int
    study_type: StudyType
    section: str | None = None
    page: int | None = None
    market: list[str] = Field(default_factory=list)
    topic: list[str] = Field(default_factory=list)
    date: str | None = None
    source_path: str
    content_hash: str


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(DocumentChunk):
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    document_title: str
    document_id: str
    version: int
    page_or_section: str | None = None
    study_type: StudyType
    supporting_excerpt: str


class GroundedAnswer(BaseModel):
    answer: str
    citation_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class EvidenceGrade(BaseModel):
    sufficient: bool
    rationale: str
    missing_evidence: list[str] = Field(default_factory=list)


class RiskEvidence(BaseModel):
    document_id: str
    version: int
    citation_id: str | None = None


class RiskFinding(BaseModel):
    category: Literal["scope", "schedule", "quality", "dependency", "approval", "data"]
    severity: Literal["low", "medium", "high", "critical"]
    statement: str
    evidence: list[RiskEvidence]
    impacted_items: list[str] = Field(default_factory=list)
    recommended_action: str
    requires_human_decision: bool = True


class EvaluationCase(BaseModel):
    id: str
    question: str
    category: str
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_answer_facts: list[str] = Field(default_factory=list)
    should_refuse: bool = False


class EvaluationResult(BaseModel):
    case_id: str
    question: str
    retrieved_document_ids: list[str]
    expected_document_ids: list[str]
    retrieval_hit: bool
    answer: str
    refused: bool
    refusal_correct: bool
    citation_ids: list[str]
    latency_seconds: float
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
