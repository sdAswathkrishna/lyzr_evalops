# Phase 1 — LangChain + LangSmith: Learn by Building

> **Goal**: Understand Traces + EvalOps by building and operating a real agent.
> Everything you learn here maps directly to what you'll build on top of Lyzr.

---

## The Agent We Build

A **Tech Support Agent** for a SaaS product with 3 tools:
- `search_knowledge_base` — looks up help articles
- `get_ticket_status` — checks an existing support ticket
- `create_ticket` — files a new support ticket

Simple enough to understand quickly. Real enough to demonstrate production patterns.

---

## Setup

### 1. Create the conda environment
```bash
cd lyzr_comparison
/opt/homebrew/Caskroom/miniconda/base/bin/conda env create -f environment.yml
/opt/homebrew/Caskroom/miniconda/base/bin/conda activate lyzr-evalops
```

### 2. Configure API keys
```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, LANGCHAIN_API_KEY
```

### 3. Run steps in order
```bash
python phase1_langsmith/01_basic_agent.py       # Step 1
python phase1_langsmith/02_langsmith_tracing.py # Step 2
python phase1_langsmith/03_create_dataset.py    # Step 3
python phase1_langsmith/04_run_evaluations.py   # Step 4
python phase1_langsmith/05_compare_versions.py  # Step 5
```

---

## What Each Step Teaches

| Step | File | Core Concept |
|------|------|--------------|
| 1 | `01_basic_agent.py` | ReAct agent: Thought → Action → Observation loop |
| 2 | `02_langsmith_tracing.py` | Tracing: spans, metadata, feedback |
| 3 | `03_create_dataset.py` | Traces → Datasets: the EvalOps foundation |
| 4 | `04_run_evaluations.py` | LLM-as-a-judge: 3 custom evaluators |
| 5 | `05_compare_versions.py` | Version comparison + release decision gate |

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **Trace** | One end-to-end execution of the agent (one user query = one trace) |
| **Span / Run** | One node inside a trace (one LLM call, one tool call) |
| **Project** | A named bucket grouping related traces in LangSmith |
| **Dataset** | A curated set of `(input, reference_output)` pairs for evaluation |
| **Example** | One row in a dataset |
| **Feedback** | A score attached to a run (human thumbs up/down or LLM judge score) |
| **Experiment** | One evaluation run: agent + dataset + evaluators → scores |
| **LLM-as-a-Judge** | Using an LLM to score another LLM's output (vs string matching) |
| **Release Gate** | A score threshold that must be passed before promoting to production |

---

## The EvalOps Loop (what you're building for Lyzr)

```
Production Traffic
      │
      ▼
  Traces (LangSmith / Lyzr)
      │
      ▼
  Dataset Curation
  (select interesting traces, add reference answers)
      │
      ▼
  Evaluation
  (LLM-as-a-judge: correctness, helpfulness, conciseness)
      │
      ▼
  Version Comparison
  (v1 vs v2 on same dataset)
      │
      ▼
  Release Decision
  PROMOTE → CANARY → HOLD → ROLLBACK
```

---

## What LangSmith Provides

After running all 5 steps, you will have seen:

- ✅ **Automatic tracing** — every LangChain call traced without code changes
- ✅ **Span tree** — full Thought/Action/Observation breakdown per trace
- ✅ **Metadata & tags** — attach user IDs, versions, environments to runs
- ✅ **Feedback API** — log human or automated scores per run
- ✅ **Datasets** — create, manage, and version evaluation datasets
- ✅ **Experiments** — run agent against dataset, store scores per example
- ✅ **Comparison UI** — side-by-side diff of two experiment runs
- ✅ **Custom evaluators** — write Python functions to score any dimension

---

## What Lyzr Has vs What It Lacks

> This is what Phase 2 will formally audit. Initial hypothesis:

| Capability | LangSmith | Lyzr (current) |
|-----------|-----------|----------------|
| Agent tracing | ✅ Full span tree | ✅ Has traces |
| Token / latency per span | ✅ | ❓ To audit |
| Feedback API | ✅ | ❓ To audit |
| Dataset creation from traces | ✅ | ❌ Likely missing |
| LLM-as-a-judge evaluations | ✅ | ❌ Likely missing |
| Experiment comparison UI | ✅ | ❌ Likely missing |
| Release decision gate | ✅ (manual) | ❌ Missing |

> **Phase 3 goal**: Build the bottom 4 rows as a service on top of Lyzr.
