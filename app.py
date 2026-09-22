from pathlib import Path

import streamlit as st

from research_assistant.config import get_settings
from research_assistant.graph.workflow import build_rag_graph
from research_assistant.providers import build_provider
from research_assistant.retrieval.index import HybridIndex
from research_assistant.risk import detect_project_risks

st.set_page_config(page_title="ConnectTel Research Assistant", page_icon="🔎", layout="wide")


@st.cache_resource
def load_runtime():
    settings = get_settings()
    provider = build_provider(settings)
    index = HybridIndex(settings, provider)
    graph = build_rag_graph(settings, provider, index)
    return settings, provider, index, graph


st.title("AI Research Assistant")
st.caption("Ask cited questions about the fictional ConnectTel research project")

if not Path("storage/chunks.json").exists():
    st.error("The corpus has not been indexed. Run `uv run python scripts/ingest.py` first.")
    st.stop()

try:
    settings, provider, index, graph = load_runtime()
except Exception as exc:
    st.error(f"Runtime configuration error: {exc}")
    st.stop()

with st.sidebar:
    st.header("Runtime")
    st.write(f"Provider: `{settings.model_provider}`")
    st.write(f"Chat: `{settings.active_chat_model}`")
    st.write(f"Embeddings: `{settings.active_embedding_model}`")
    st.write(f"Reranking: `{'enabled' if settings.enable_rerank else 'disabled'}`")
    debug = st.toggle("Show retrieval details", value=True)

tab_qa, tab_risk = st.tabs(["Ask the Project", "Scope Risk Scan"])

with tab_qa:
    with st.expander("Example questions"):
        st.markdown(
            """
- What is the primary objective of the study?
- Which markets are currently approved?
- Has Spain been approved as an additional market?
- Which historical billing questions could be considered for reuse?
- What additional budget was approved for Spain?
"""
        )
    question = st.text_input(
        "Question",
        placeholder="Is Spain part of the approved project scope?",
    )
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving and checking evidence..."):
            state = graph.invoke({"question": question})
        rerank_warning = getattr(provider, "rerank_warning", None)
        if rerank_warning:
            st.warning(rerank_warning)
        st.subheader("Answer")
        st.write(state["answer"])
        if state.get("missing_evidence"):
            st.info("Missing evidence: " + "; ".join(state["missing_evidence"]))

        citations = {citation.citation_id: citation for citation in state.get("citations", [])}
        used = state.get("used_citation_ids", [])
        if used:
            st.subheader("Sources")
            for citation_id in used:
                citation = citations[citation_id]
                label = (
                    f"[{citation_id}] {citation.document_title} - v{citation.version} - "
                    f"{citation.page_or_section or 'document'}"
                )
                with st.expander(label):
                    st.caption(f"Type: {citation.study_type} | ID: {citation.document_id}")
                    st.write(citation.supporting_excerpt)

        if debug:
            st.subheader("Retrieval details")
            rows = []
            for chunk in state.get("evidence", []):
                rows.append(
                    {
                        "document": chunk.metadata.document_title,
                        "type": chunk.metadata.study_type,
                        "section": chunk.metadata.section,
                        "dense": chunk.dense_score,
                        "BM25": chunk.sparse_score,
                        "fusion": chunk.fusion_score,
                        "rerank": chunk.rerank_score,
                    }
                )
            st.dataframe(rows, use_container_width=True)

with tab_risk:
    st.subheader("Optional project review")
    st.write(
        "This scan is separate from question answering. Run it only when you want to compare "
        "scope, approval, questionnaire, sample-plan, timeline, and historical evidence."
    )
    if st.button("Run Scope Risk Scan", type="primary"):
        diagnostics: list[str] = []
        with st.spinner("Comparing project artifacts..."):
            findings = detect_project_risks(provider, index, diagnostics=diagnostics)
        rerank_warning = getattr(provider, "rerank_warning", None)
        if rerank_warning:
            st.warning(rerank_warning)
        for diagnostic in diagnostics:
            st.warning(diagnostic)
        if not findings and diagnostics:
            st.warning("The scan is inconclusive; it does not establish that there are no risks.")
        elif not findings:
            st.info("The model returned no risks from the retrieved evidence.")
        for finding in findings:
            st.subheader(f"{finding.severity.upper()}: {finding.category}")
            st.write(finding.statement)
            st.write("**Recommended action:** " + finding.recommended_action)
            st.write("**Impacted items:** " + ", ".join(finding.impacted_items))
            st.caption(
                "Human decision required: " + ("Yes" if finding.requires_human_decision else "No")
            )
            st.json([item.model_dump() for item in finding.evidence])
