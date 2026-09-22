# AI Research Assistant

A focused Hybrid RAG application for a fictional market-research project. It answers questions from project documents with citations, refuses unsupported questions, and provides a separate user-triggered **Scope Risk Scan**.

All ConnectTel documents in this repository are synthetic. They do not describe a real company or client engagement.

## What the application does

- Answers questions about objectives, scope, methodology, deliverables, dates, and decisions.
- Searches both current-project documents and selected historical research.
- Clearly distinguishes historical learning from current-project facts.
- Shows the source document, version, section, and supporting excerpt.
- Refuses when the available documents do not contain enough evidence.
- Runs scope-risk detection only when the user clicks the separate button.
- Includes 15 fixed evaluation questions and deterministic unit tests.

## Simple architecture

### Document preparation

```text
Documents
  -> LlamaParse or local loaders
  -> structure-aware chunks with metadata
  -> Nebius embeddings
  -> Chroma vector index + local BM25 index
```

### Question answering

```text
Question
  -> vector search + BM25 keyword search
  -> Reciprocal Rank Fusion
  -> evidence check
  -> cited answer OR grounded refusal
```

The LangGraph workflow is intentionally small:

1. `classify_query` adds a traceable question label.
2. `retrieve` runs Hybrid RAG and prepares citations.
3. `grade_evidence` recognizes direct negative approval or scope statements before asking the model about other evidence. The deterministic check requires the question subject and its negative status to be linked in the same sentence of a current-project source.
4. `generate_answer` returns a cited answer when evidence is sufficient.
5. `refuse` returns a standard response when evidence is insufficient.

## What Hybrid RAG means

RAG retrieves source material before asking a language model to answer. **Hybrid RAG** uses two searches over the same chunks:

- **Vector search** finds similar meaning. It can connect “geographic expansion” with a passage describing an “additional study market,” even when the wording differs.
- **BM25 keyword search** finds exact terms such as `Spain`, `October 20`, document versions, or questionnaire numbers.
- **Reciprocal Rank Fusion** combines the two ranked lists without trying to compare their incompatible raw scores.

For the Spain scenario, vector search helps find passages about scope expansion, while BM25 finds exact references to Spain, approval, budget, routing, and sample quotas.

## Approaches considered

| Approach | How it works | Advantages | Limitations | Fit for this project |
|---|---|---|---|---|
| Simple Vector RAG | Runs semantic vector retrieval, then sends the results directly to the model. | Fastest to build and easy to explain. | Can miss exact names, dates, IDs, and questionnaire items; no explicit answer-or-refuse gate. | Acceptable for a basic prototype, but weaker for precise project facts. |
| **Hybrid RAG + small LangGraph** | Combines vector and BM25 retrieval, fuses results, checks evidence, then answers or refuses. | Better coverage of meaning and exact terms; predictable flow; citations and refusal are explicit. | Adds a small amount of implementation effort and one evidence-grading model call. | **Selected. Best balance of quality, simplicity, and explainability.** |
| Full Agentic RAG | Lets an agent rewrite queries, choose tools, repeat searches, and decide when to stop. | Useful for open-ended research across many tools or external sources. | More variable cost, latency, behavior, and debugging effort. | Unnecessary for a fixed class-project knowledge base. |

## Technology choices

| Component | Use in this project |
|---|---|
| LlamaParse | Parses complex PDF, DOCX, PPTX, and XLSX files while retaining useful structure. The included Markdown, JSON, and CSV files use local loaders. |
| Nebius text model | Grades evidence, generates cited answers, and creates structured risk findings. |
| Nebius embeddings | Converts document chunks and questions into vectors for semantic search. |
| Chroma | Stores vectors and metadata locally. |
| BM25 | Finds exact terminology, names, dates, markets, and IDs. |
| Reciprocal Rank Fusion | Combines vector and BM25 rankings. |
| LangGraph | Controls the bounded retrieve, grade, answer-or-refuse workflow. |
| Streamlit | Provides the Ask the Project and Scope Risk Scan tabs. |

Reranking is optional and disabled by default because the Nebius `/v1/rerank` endpoint may not be available to every account. The application works without it.

## Scope Risk Scan

Risk detection is a separate feature; it is not run after every question. The scan retrieves and compares evidence about:

- scope-change requests and approval status;
- questionnaire and sample-plan mismatches;
- schedule dependencies and deadline conflicts;
- relevant historical learning missing from the current questionnaire.

The synthetic corpus includes a request to consider Spain as an additional market. The request is deliberately **not approved**. The questionnaire, sample plan, budget, and schedule have not been updated. The detector must report those gaps without treating the request as an approval.

Approval questions have a deterministic safeguard. It runs only for approval, inclusion, and scope questions; ignores historical sources; and requires a direct subject-status relationship in one sentence. For example, “Spain has not been approved” qualifies, while “France is approved, while Spain is not approved” does not supply negative evidence about France. A request by itself is also not treated as approval evidence. These checks prevent incorrect refusals without assigning one market's status to another.

## Repository structure

```text
ai-research-assistant/
├── app.py
├── pyproject.toml
├── .env.example
├── data/
│   ├── corpus/
│   │   ├── current_project/
│   │   ├── historical_studies/
│   │   └── manifest.json
│   └── eval_questions.json
├── docs/
│   └── AI_Research_Assistant_Solution_Design.docx
├── scripts/
│   ├── ingest.py
│   ├── list_nebius_models.py
│   ├── smoke_test_providers.py
│   └── run_evaluation.py
├── src/research_assistant/
│   ├── graph/
│   ├── ingestion/
│   ├── retrieval/
│   ├── citations.py
│   ├── config.py
│   ├── providers.py
│   ├── risk.py
│   └── schemas.py
├── tests/
└── reports/
```

## Setup on macOS

### 1. Install prerequisites

```bash
brew install git uv python@3.12
brew install --cask visual-studio-code
```

Verify them:

```bash
git --version
uv --version
python3 --version
code --version
```

### 2. Create the environment

```bash
cd ai-research-assistant
uv sync
cp .env.example .env
```

Add your credentials to `.env`. Do not commit or paste that file into chat.

### 3. Select and test provider models

Your LlamaParse key is the LlamaCloud project key used by this project.

```bash
uv run python scripts/list_nebius_models.py
uv run python scripts/smoke_test_providers.py
```

In Nebius:

- **Text-to-Text** is used for chat/generation.
- **Embedding** is used for vectors.
- **Vision** is not needed for this MVP.
- Reranking is optional. Keep `ENABLE_RERANK=false` unless its smoke test succeeds.

### 4. Build the indexes

```bash
uv run python scripts/ingest.py
```

The included files are parsed locally. LlamaParse is used when you add PDF, DOCX, PPTX, or XLSX files and configure `USE_LLAMAPARSE=true`.

### 5. Start the app

```bash
uv run streamlit run app.py
```

### 6. Test and evaluate

```bash
uv run pytest
uv run ruff check .
uv run python scripts/run_evaluation.py
```

Evaluation results are written to:

```text
reports/evaluation_results.csv
reports/evaluation_results.json
```

## Suggested demonstration

1. Ask: “What is the primary objective of the study?”
2. Ask: “Has Spain been approved as an additional market?”
3. Ask: “Which historical billing questions could be considered for reuse?”
4. Ask: “What additional budget was approved for Spain?” to show safe refusal.
5. Open the separate Scope Risk Scan tab and run the scan.
6. Show vector, BM25, and fusion scores in retrieval details.

## Evaluation and safeguards

Measure actual results for retrieval hit rate, citation accuracy, answer faithfulness, refusal accuracy, risk support, and response time. Do not present a target accuracy as an achieved score.

- Never put real confidential project information in the demo corpus.
- Treat project documents—not memory or model output—as the source of truth.
- Do not convert a request into an approval.
- Label historical findings as historical evidence.
- Keep `.env`, generated indexes, and evaluation outputs out of Git.
- Cache parsed content and embeddings when extending the project to conserve credits.

## MVP boundaries

The project intentionally excludes autonomous web search, unbounded agent loops, automatic project-plan changes, client communication, authentication, and production deployment. Mem0 may be added later for user formatting preferences, but it must not become a source of project facts.
