import csv
import json
import time
from pathlib import Path

from research_assistant.config import get_settings
from research_assistant.graph.workflow import build_rag_graph
from research_assistant.providers import build_provider
from research_assistant.retrieval.index import HybridIndex
from research_assistant.schemas import EvaluationCase, EvaluationResult


def main() -> None:
    settings = get_settings()
    provider = build_provider(settings)
    index = HybridIndex(settings, provider)
    graph = build_rag_graph(settings, provider, index)
    cases = [
        EvaluationCase.model_validate(item)
        for item in json.loads(Path("data/eval_questions.json").read_text(encoding="utf-8"))
    ]
    results: list[EvaluationResult] = []

    for case in cases:
        started = time.perf_counter()
        try:
            state = graph.invoke({"question": case.question})
            retrieved = list(
                dict.fromkeys(chunk.metadata.document_id for chunk in state.get("evidence", []))
            )
            expected = set(case.expected_document_ids)
            result = EvaluationResult(
                case_id=case.id,
                question=case.question,
                retrieved_document_ids=retrieved,
                expected_document_ids=case.expected_document_ids,
                retrieval_hit=expected.issubset(set(retrieved)),
                answer=state["answer"],
                refused=bool(state.get("refused", False)),
                refusal_correct=bool(state.get("refused", False)) == case.should_refuse,
                citation_ids=state.get("used_citation_ids", []),
                latency_seconds=round(time.perf_counter() - started, 3),
            )
        except Exception as exc:
            result = EvaluationResult(
                case_id=case.id,
                question=case.question,
                retrieved_document_ids=[],
                expected_document_ids=case.expected_document_ids,
                retrieval_hit=False,
                answer="",
                refused=False,
                refusal_correct=False,
                citation_ids=[],
                latency_seconds=round(time.perf_counter() - started, 3),
                error=str(exc),
            )
        results.append(result)
        print(f"{case.id}: retrieval={result.retrieval_hit} refusal={result.refusal_correct}")

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "evaluation_results.json").write_text(
        json.dumps([result.model_dump() for result in results], indent=2), encoding="utf-8"
    )
    with (output_dir / "evaluation_results.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "category",
            "retrieval_hit",
            "refusal_correct",
            "refused",
            "latency_seconds",
            "retrieved_document_ids",
            "expected_document_ids",
            "citation_ids",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        category_by_id = {case.id: case.category for case in cases}
        for result in results:
            writer.writerow(
                {
                    "case_id": result.case_id,
                    "category": category_by_id[result.case_id],
                    "retrieval_hit": result.retrieval_hit,
                    "refusal_correct": result.refusal_correct,
                    "refused": result.refused,
                    "latency_seconds": result.latency_seconds,
                    "retrieved_document_ids": ";".join(result.retrieved_document_ids),
                    "expected_document_ids": ";".join(result.expected_document_ids),
                    "citation_ids": ";".join(result.citation_ids),
                    "error": result.error or "",
                }
            )

    hit_rate = sum(item.retrieval_hit for item in results) / len(results)
    refusal_rate = sum(item.refusal_correct for item in results) / len(results)
    print(f"Retrieval exact-source-set rate: {hit_rate:.1%}")
    print(f"Refusal classification accuracy: {refusal_rate:.1%}")


if __name__ == "__main__":
    main()
