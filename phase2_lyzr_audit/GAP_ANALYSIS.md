# Phase 2 — Lyzr vs LangSmith: Full Gap Analysis

> **Generated from live SDK inspection + docs audit — April 2026**

---

## What We Did

1. **Built the same Tech Support Agent in both platforms** — same tools, same queries, different SDKs
2. **Inspected every public method** on `lyzr.Studio` (41 methods) and `lyzr_agent_api.AgentAPI` (27 methods)
3. **Attempted 5 EvalOps calls** that should exist if Lyzr had LangSmith parity — all 5 failed
4. **Cross-referenced Lyzr docs** for any features not exposed in the SDK

---

## SDK Facts (from live inspection)

### `lyzr.Studio` — 41 public methods

**What it has (works well):**
```
acreate_agent / aupdate_agent / adelete_agent / aget_agent / alist_agents
acreate_knowledge_base / adelete_knowledge_base / aget_knowledge_base / alist_knowledge_bases
acreate_rai_policy / aget_rai_policy / alist_rai_policies
acreate_memory_credential / aget_memory / alist_memories / aget_memory_messages
acreate_schedule / adelete_schedule / alist_schedules / atrigger_schedule
acreate_context / aget_context / alist_contexts
```

**What it is completely missing (0 methods):**
```
Tracing:    0 methods   (no get_trace, list_runs, get_span, export_traces)
Feedback:   0 methods   (no add_feedback, score_run, create_feedback)
Dataset:    0 methods   (no create_dataset, add_example, list_datasets)
Evaluation: 0 methods   (no evaluate, run_evaluator, create_experiment)
```

### `lyzr_agent_api.AgentAPI` — 27 methods

Covers: `chat_with_agent`, `create_agent_endpoint`, `create_environment_endpoint`,
`get_activities_by_user_session`, `get_session_history`, `get_task_status`, etc.

**EvalOps methods present: 0 / 12 tested**
```
get_trace          ✗   list_traces        ✗   get_run           ✗   list_runs         ✗
add_feedback       ✗   create_feedback    ✗   score_run         ✗
create_dataset     ✗   list_datasets      ✗   add_example       ✗
run_evaluation     ✗   create_experiment  ✗
```

---

## Full Feature Matrix

| # | Feature | LangSmith | Lyzr | Gap |
|---|---------|-----------|------|-----|
| 1 | Automatic tracing of agent runs | ✅ Full | ✅ Full | None |
| 2 | Per-span details (prompt text, LLM response body) | ✅ Full | ⚠️ Partial | Small |
| 3 | Token counts per span | ✅ Full | ✅ Full | None |
| 4 | Latency per span | ✅ Full | ✅ Full | None |
| 5 | **Programmatic trace access (list/get runs via SDK)** | ✅ Full | ❌ Missing | **CRITICAL** |
| 6 | Trace metadata & tags (user_id, version, env) | ✅ Full | ⚠️ Partial | Medium |
| 7 | **Feedback API (attach score to a run)** | ✅ Full | ❌ Missing | **CRITICAL** |
| 8 | **Multi-dimensional feedback (correctness, helpfulness)** | ✅ Full | ❌ Missing | **CRITICAL** |
| 9 | **Evaluation dataset management (create, add, version)** | ✅ Full | ❌ Missing | **CRITICAL** |
| 10 | **Traces → Dataset pipeline** | ✅ Full | ❌ Missing | **CRITICAL** |
| 11 | Built-in safety evaluators (toxicity, PII, hallucination) | ❌ Missing | ✅ Full | Lyzr wins |
| 12 | **Custom LLM-as-a-judge evaluators** | ✅ Full | ❌ Missing | **CRITICAL** |
| 13 | **Criteria-based evaluation (no reference needed)** | ✅ Full | ❌ Missing | HIGH |
| 14 | **Reference-based evaluation (compare to ground truth)** | ✅ Full | ❌ Missing | HIGH |
| 15 | **Experiment runner (agent × dataset → scores)** | ✅ Full | ❌ Missing | **CRITICAL** |
| 16 | **Experiment comparison (v1 vs v2 on same dataset)** | ✅ Full | ❌ Missing | **CRITICAL** |
| 17 | **Release decision gate (PROMOTE/CANARY/HOLD/ROLLBACK)** | ⚠️ Manual | ❌ Missing | HIGH |
| 18 | Canary deployment (route X% traffic to new version) | ❌ Missing | ❌ Missing | Both need it |
| 19 | Agent version rollback | ❌ Missing | ⚠️ Partial | Medium |
| 20 | Auto-generated test cases (persona × scenario matrix) | ❌ Missing | ✅ Full | Lyzr wins |

**Gap count: 8 CRITICAL, 3 HIGH, 2 Medium, 2 Small, 3 No gap, 3 Lyzr wins**

---

## What Lyzr Does Better Than LangSmith

These are **real advantages** — do not rebuild them:

| Lyzr Strength | LangSmith Equivalent | Verdict |
|--------------|---------------------|---------|
| Built-in RAI: toxicity, PII (9 types), hallucination, injection | None | **Keep Lyzr's** |
| Trained ML safety classifiers (not LLM judge) | None | **Keep Lyzr's** |
| `create_agent(memory=10)` — memory in one line | ConversationBufferMemory + chain wiring | **Lyzr simpler** |
| Multi-LLM with one param: `provider='anthropic'` | Separate import per provider | **Lyzr simpler** |
| No-code Studio builder for non-technical users | Developer-only | **Unique to Lyzr** |
| Agent Simulation Engine (persona × scenario → test cases) | Manual test writing | **Keep Lyzr's** |

---

## The 7 Things to Build in Phase 3

> **Build order = priority order.** Each item depends on the one before it.

### #1 — Trace Access API *(Medium effort)*
**What:** A thin wrapper around Lyzr's internal OTel traces that exposes them as Python objects.
**Why first:** Everything downstream (dataset curation, eval, comparison) requires programmatic access to runs.
**Interface:**
```python
from evalops import TraceClient
client = TraceClient(lyzr_api_key="...")
runs = client.list_runs(project="my-project", limit=50)
run  = client.get_run(run_id="abc123")
# run.inputs, run.outputs, run.latency_ms, run.token_count
```
**Maps to:** `langsmith.Client.list_runs()` + `get_run()`

---

### #2 — Feedback API *(Small effort)*
**What:** REST endpoint + Python client to attach named scores to a Lyzr run by ID.
**Why:** Without scores, you can't separate good runs from bad ones. No signal = no EvalOps.
**Interface:**
```python
client.add_feedback(
    run_id="abc123",
    key="correctness",     # any dimension name
    score=0.9,
    comment="Factually correct, missing one detail"
)
```
**Maps to:** `langsmith.Client.create_feedback(run_id, key, score)`

---

### #3 — Dataset Store *(Medium effort)*
**What:** A versioned store for `(input, reference_output, metadata)` examples.
**Why:** The "answer key" every evaluation compares against. Can't grade without it.
**Interface:**
```python
dataset = client.create_dataset("tech-support-v1")
client.add_example(dataset.id, inputs={"input": "..."}, outputs={"output": "..."})
examples = client.list_examples(dataset.id)
```
**Maps to:** `langsmith.Client.create_dataset()` + `create_example()`

---

### #4 — Trace → Dataset Pipeline *(Medium effort)*
**What:** UI flow + API to save a production trace directly as a dataset example.
**Why:** Real users generate edge cases you can't write manually. Closes the prod → eval loop.
**Interface:**
```python
# API: promote a production run to an eval example
client.add_run_to_dataset(
    run_id="abc123",
    dataset_id=dataset.id,
    reference_output={"output": "corrected ideal answer"}  # optional
)
```
**UI:** "Save to Dataset" button on every trace row in Studio.
**Maps to:** LangSmith's "Add to Dataset" trace action

---

### #5 — Custom Evaluator Registry *(Medium effort)*
**What:** A way for users to register Python scoring functions as named evaluators.
**Why:** Lyzr's built-in safety evals are not enough. Teams need domain-specific scoring.
**Interface:**
```python
from evalops import evaluator

@evaluator
def correctness(run, example) -> dict:
    # call an LLM judge, return score + comment
    return {"score": 0.9, "comment": "..."}

client.register_evaluator(correctness)
```
**Maps to:** LangSmith's `run_evaluator` decorator + custom evaluator functions in `evaluate()`

---

### #6 — Experiment Runner *(Medium effort)*
**What:** Run a Lyzr agent against a dataset with one or more evaluators, store scores per example.
**Why:** Turns dataset + evaluators into a numbered, reproducible experiment.
**Interface:**
```python
from evalops import run_experiment

results = run_experiment(
    agent_fn=my_agent_runner,
    dataset="tech-support-v1",
    evaluators=["correctness", "helpfulness"],
    experiment_name="agent-v2-improved-prompt",
)
```
**Maps to:** `langsmith.evaluation.evaluate()`

---

### #7 — Experiment Comparison + Release Decision *(Small effort)*
**What:** Side-by-side score delta for two experiments + structured PROMOTE/CANARY/HOLD/ROLLBACK output.
**Why:** This is the entire product — converting numbers into a team decision.
**Interface:**
```python
from evalops import compare_experiments

decision = compare_experiments("agent-v1", "agent-v2")
# decision.verdict  → "CANARY"
# decision.deltas   → {"correctness": +0.12, "helpfulness": -0.01}
# decision.reason   → "v2 improves 2/3 metrics with no regressions"
```
**Maps to:** LangSmith's Compare Experiments UI (we built the logic in `05_compare_versions.py`)

---

## Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════╗
║          YOUR PRODUCT  (Traces + EvalOps layer)                  ║
║                                                                  ║
║  #1 Trace Access  →  #2 Feedback API  →  #3 Dataset Store        ║
║         │                   │                   │                ║
║         └───────────────────┴───────────────────┘                ║
║                             │                                    ║
║                  #4 Trace → Dataset Pipeline                     ║
║                             │                                    ║
║                  #5 Custom Evaluator Registry                    ║
║                             │                                    ║
║                    #6 Experiment Runner                          ║
║                             │                                    ║
║         #7 Experiment Comparison + Release Decision              ║
║              PROMOTE | CANARY | HOLD | ROLLBACK                  ║
╠══════════════════════════════════════════════════════════════════╣
║          LYZR PLATFORM  (existing — do not rebuild)              ║
║  Agent Builder │ Workflows │ Governance │ Memory                  ║
║  RAI Guardrails│ OTel Traces│ Studio UI │ A-Sim Engine            ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Phase 3 Effort Estimate

| Item | Effort | Dependency |
|------|--------|-----------|
| #1 Trace Access API | ~3 days | None |
| #2 Feedback API | ~1 day | #1 |
| #3 Dataset Store | ~2 days | None (parallel with #2) |
| #4 Trace → Dataset Pipeline | ~2 days | #1 + #3 |
| #5 Custom Evaluator Registry | ~2 days | #3 |
| #6 Experiment Runner | ~3 days | #4 + #5 |
| #7 Comparison + Decision Engine | ~2 days | #6 |
| **Total** | **~15 days** | |

> A working MVP of the full EvalOps loop is **~3 weeks of focused engineering.**
