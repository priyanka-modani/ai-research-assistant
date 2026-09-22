import pytest

from research_assistant.risk import detect_project_risks
from research_assistant.schemas import ChunkMetadata, RetrievedChunk


class FakeIndex:
    def search(self, query):
        return [RetrievedChunk(
            chunk_id="timeline:1",
            text="T04 fieldwork ends October 20. M02 presentation is October 20, before reporting.",
            metadata=ChunkMetadata(
                project_id="p", document_id="timeline_v1", document_title="Timeline",
                artifact_type="timeline", version=1, study_type="current_project",
                source_path="timeline.csv", content_hash="abc",
            ),
        )]


class FakeProvider:
    def __init__(self, payload):
        self.payload = payload

    def generate_json(self, system, user):
        return self.payload


def risk(citation_id="C1", document_id="timeline_v1", version=1):
    return {
        "category": "schedule", "severity": "critical",
        "statement": "Presentation precedes reporting.",
        "evidence": [{"citation_id": citation_id, "document_id": document_id, "version": version}],
        "recommended_action": "Agree a revised schedule.", "requires_human_decision": True,
    }


@pytest.mark.parametrize("citation_id", ["T04", "T05", "T06", "M02", "timeline_v1"])
def test_task_and_document_ids_are_reported_not_silently_discarded(citation_id):
    diagnostics = []
    findings = detect_project_risks(
        FakeProvider({"risks": [risk(citation_id)]}), FakeIndex(), diagnostics=diagnostics,
    )
    assert findings == []
    assert any("no valid evidence remained" in message for message in diagnostics)


def test_valid_schedule_risk_survives():
    diagnostics = []
    findings = detect_project_risks(
        FakeProvider({"risks": [risk()]}), FakeIndex(), diagnostics=diagnostics,
    )
    assert len(findings) == 1
    assert findings[0].evidence[0].citation_id == "C1"
    assert diagnostics == []


@pytest.mark.parametrize("change", [{"document_id": "wrong"}, {"version": 2}])
def test_mismatched_source_metadata_is_rejected(change):
    diagnostics = []
    assert detect_project_risks(
        FakeProvider({"risks": [risk(**change)]}), FakeIndex(), diagnostics=diagnostics,
    ) == []
    assert diagnostics


@pytest.mark.parametrize("payload", [{}, {"risks": None}, {"risks": [{}]}])
def test_malformed_response_is_not_treated_as_no_risks(payload):
    diagnostics = []
    assert detect_project_risks(
        FakeProvider(payload), FakeIndex(), diagnostics=diagnostics,
    ) == []
    assert diagnostics


def test_explicit_empty_result_has_no_validation_warning():
    diagnostics = []
    assert detect_project_risks(
        FakeProvider({"risks": []}), FakeIndex(), diagnostics=diagnostics,
    ) == []
    assert diagnostics == []


def test_invalid_finding_does_not_hide_valid_findings():
    diagnostics = []
    findings = detect_project_risks(
        FakeProvider({"risks": [risk("T04"), risk()]}),
        FakeIndex(), diagnostics=diagnostics,
    )
    assert len(findings) == 1
    assert diagnostics


class RepairProvider:
    def __init__(self, corrected_reference):
        self.calls = []
        self.corrected_reference = corrected_reference

    def generate_json(self, system, user):
        self.calls.append((system, user))
        assert len(self.calls) <= 2
        if len(self.calls) == 1:
            return {"risks": [risk("T04"), risk()]}
        assert '"citation_id": "C1"' in user
        assert "T04 fieldwork" in user
        return {"corrections": [{
            "risk_index": 0, "evidence": [self.corrected_reference],
            "statement": "This must not replace the original statement.",
        }]}


def test_citation_repair_recovers_risk_without_changing_claims_or_valid_findings():
    provider = RepairProvider(risk()["evidence"][0])
    diagnostics = []
    findings = detect_project_risks(provider, FakeIndex(), diagnostics=diagnostics)
    assert len(provider.calls) == 2
    assert len(findings) == 2
    assert findings[0].statement == risk()["statement"]
    assert findings[0].evidence[0].citation_id == "C1"
    assert diagnostics == []


@pytest.mark.parametrize("reference", [
    risk("T04")["evidence"][0],
    risk(version=2)["evidence"][0],
    risk(document_id="wrong")["evidence"][0],
])
def test_failed_repair_is_bounded_and_does_not_hide_valid_risk(reference):
    provider = RepairProvider(reference)
    diagnostics = []
    findings = detect_project_risks(provider, FakeIndex(), diagnostics=diagnostics)
    assert len(provider.calls) == 2
    assert len(findings) == 1
    assert any("Rejected references:" in message for message in diagnostics)
