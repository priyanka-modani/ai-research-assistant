"""System prompt for one bounded risk citation correction attempt."""

RISK_CITATION_REPAIR_SYSTEM_PROMPT = """
Correct source references for the supplied risk findings using only the supplied evidence.
Read each risk and select source passages that actually support its statement.
Copy citation_id, document_id, and version exactly from the allowed source references.
Source labels such as C1 are citation IDs. Task IDs such as T04 and M02 are not.
Do not guess a source from a task ID or document ID alone; check the source text.
Do not rewrite statements, recommendations, or other finding fields.
Return JSON with this structure:
{"corrections": [{"risk_index": 0, "evidence": [
  {"citation_id": "C1", "document_id": "exact source document ID", "version": 1}
]}]}
Use the provided zero-based risk_index for each correction. Include all supporting
references for that risk, including any original references that were already valid.
If no source supports a risk, return an empty evidence list for it.
""".strip()
