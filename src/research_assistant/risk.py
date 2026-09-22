import json
import warnings

from pydantic import ValidationError

from research_assistant.citations import make_citations
from research_assistant.prompts.risk_citation_repair import RISK_CITATION_REPAIR_SYSTEM_PROMPT
from research_assistant.prompts.risk_detection import RISK_DETECTION_SYSTEM_PROMPT
from research_assistant.providers import ModelProvider
from research_assistant.retrieval.index import HybridIndex
from research_assistant.schemas import RiskFinding

RISK_SCAN_QUERY = (
    "Find scope changes, approval gaps, market conflicts, questionnaire mismatches, sample plan "
    "mismatches, dates, dependencies, deadlines, and historical research omissions."
)


def detect_project_risks(
    provider: ModelProvider, index: HybridIndex, *, diagnostics: list[str] | None = None
) -> list[RiskFinding]:
    def report(message: str) -> None:
        if diagnostics is not None:
            diagnostics.append(message)
        else:
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    queries = [
        RISK_SCAN_QUERY,
        "Spain request approval questionnaire routing sample quota budget market scope",
        (
            "October 20 questionnaire approval programming testing fieldwork analysis "
            "reporting schedule"
        ),
        "historical billing transparency current questionnaire missing questions quality risk",
    ]
    by_id = {}
    for query in queries:
        for chunk in index.search(query):
            by_id.setdefault(chunk.chunk_id, chunk)
    evidence = list(by_id.values())[:15]
    if not evidence:
        report("The risk scan retrieved no evidence. The scan is inconclusive.")
        return []
    citations = make_citations(evidence)
    blocks = []
    for citation, chunk in zip(citations, evidence, strict=True):
        blocks.append(
            f"[{citation.citation_id}] {citation.document_title}; "
            f"citation_id={citation.citation_id}; "
            f"document_id={citation.document_id}; "
            f"version={citation.version}; study_type={citation.study_type}\n{chunk.text}"
        )
    payload = provider.generate_json(
        system=RISK_DETECTION_SYSTEM_PROMPT,
        user="\n\n---\n\n".join(blocks),
    )
    valid_citations = {citation.citation_id: citation for citation in citations}
    findings: list[RiskFinding] = []
    if not isinstance(payload, dict) or not isinstance(payload.get("risks"), list):
        report("The model did not return the required risks list. The scan is inconclusive.")
        return []
    def valid_reference(item) -> bool:
        source = valid_citations.get(item.citation_id)
        return bool(
            source and source.document_id == item.document_id and source.version == item.version
        )

    repair_candidates = {}
    for position, raw in enumerate(payload["risks"]):
        try:
            finding = RiskFinding.model_validate(raw)
        except ValidationError:
            continue
        if not finding.evidence or any(not valid_reference(item) for item in finding.evidence):
            repair_candidates[position] = finding
    if repair_candidates:
        allowed = [
            {"citation_id": c.citation_id, "document_id": c.document_id, "version": c.version}
            for c in citations
        ]
        try:
            repair = provider.generate_json(
                system=RISK_CITATION_REPAIR_SYSTEM_PROMPT,
                user=(
                    "Allowed source references:\n" + json.dumps(allowed)
                    + "\nFindings with rejected or missing references:\n"
                    + json.dumps([
                        {"risk_index": position, "finding": finding.model_dump()}
                        for position, finding in repair_candidates.items()
                    ])
                    + "\nSource evidence:\n" + "\n\n---\n\n".join(blocks)
                ),
            )
            corrections = repair.get("corrections", []) if isinstance(repair, dict) else []
            if isinstance(corrections, list):
                for correction in corrections:
                    if not isinstance(correction, dict):
                        continue
                    position = correction.get("risk_index")
                    if type(position) is not int or position not in repair_candidates:
                        continue
                    original = repair_candidates[position]
                    try:
                        candidate = RiskFinding.model_validate({
                            **original.model_dump(), "evidence": correction.get("evidence"),
                        })
                    except ValidationError:
                        continue
                    if candidate.evidence and all(valid_reference(e) for e in candidate.evidence):
                        payload["risks"][position] = candidate.model_dump()
        except Exception:
            report("The citation correction attempt failed; checking the original references.")

    for number, raw in enumerate(payload["risks"], 1):
        try:
            finding = RiskFinding.model_validate(raw)
        except ValidationError:
            report(f"Risk {number} was discarded because its response format was invalid.")
            continue
        original_count = len(finding.evidence)
        finding.evidence = [
            item
            for item in finding.evidence
            if valid_reference(item)
        ]
        removed = original_count - len(finding.evidence)
        if removed:
            report(
                f"Risk {number} ({finding.category}): rejected {removed} evidence reference(s) "
                "because the citation ID, document ID, or version did not match a supplied source."
            )
            rejected = [
                item for item in raw["evidence"]
                if item not in [entry.model_dump() for entry in finding.evidence]
            ]
            report(f"Rejected references: {json.dumps(rejected)}")
        if finding.evidence:
            findings.append(finding)
        else:
            report(
                f"Risk {number} ({finding.category}) was discarded: no valid evidence remained."
            )
    return findings
