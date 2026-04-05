"""
api.py — FastAPI application: all routes, page handlers, and REST endpoints.

Structure:
  Page routes  (HTML)  → /  /traces  /datasets  /experiments  /comparison  /chat
  Data API     (JSON)  → /api/summary  /api/traces  /api/experiments
  Chat API     (JSON)  → POST /api/chat
  EvalOps API  (JSON)  → POST /api/traces/{id}/add-to-dataset
                         POST /api/datasets/{id}/run-eval
                         POST /api/traces/{id}/feedback
"""

import os, sys, json, uuid
from datetime import datetime

# ── Path bootstrap ────────────────────────────────────────────────────────────
# File lives at: phase3_poc/backend/api.py
# parents[2] = lyzr_comparison/ ← ROOT
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from phase3_poc.backend.db import BridgeDB
from phase3_poc.backend.evaluator import EVALUATORS

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Lyzr EvalOps POC")

# Templates live in ui/templates/ (one level up from backend/, then into ui/)
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "ui", "templates")
templates     = Jinja2Templates(directory=TEMPLATES_DIR)
db            = BridgeDB()
SUMMARY_PATH  = os.path.join(os.path.dirname(__file__), "..", "poc_summary.json")

# ── Lazy agent loader ─────────────────────────────────────────────────────────
_agent = None

def get_chat_agent():
    global _agent
    if _agent is None:
        from phase3_poc.agents.langchain_agent.agent import get_agent
        _agent = get_agent(db=db, agent_version="v1-live")
    return _agent


# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    user_id:    str = "anon"
    session_id: str | None = None

class AddToDatasetRequest(BaseModel):
    dataset_name: str = "production-chat-traces"

class FeedbackRequest(BaseModel):
    score:   float          # 1.0 = positive, 0.0 = negative
    comment: str = ""
    key:     str = "user_feedback"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_summary():
    try:
        with open(SUMMARY_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={
        "summary": load_summary(),
        "active":  "home",
    })

@app.get("/traces", response_class=HTMLResponse)
async def traces(request: Request):
    return templates.TemplateResponse(request=request, name="traces.html", context={
        "lyzr_traces": db.list_traces(platform="lyzr",      limit=50),
        "ls_traces":   db.list_traces(platform="langsmith",  limit=50),
        "active":      "traces",
    })

@app.get("/datasets", response_class=HTMLResponse)
async def datasets(request: Request):
    all_datasets = db.list_datasets()
    examples     = db.list_examples(all_datasets[0]["id"]) if all_datasets else []
    return templates.TemplateResponse(request=request, name="datasets.html", context={
        "datasets": all_datasets,
        "examples": examples,
        "active":   "datasets",
    })

@app.get("/experiments", response_class=HTMLResponse)
async def experiments(request: Request):
    exp_data = [
        {**exp, "scores": db.get_aggregate_scores(exp["id"])}
        for exp in db.list_experiments()
    ]
    return templates.TemplateResponse(request=request, name="experiments.html", context={
        "experiments": exp_data,
        "active":      "experiments",
    })

@app.get("/comparison", response_class=HTMLResponse)
async def comparison(request: Request):
    return templates.TemplateResponse(request=request, name="comparison.html", context={
        "summary": load_summary(),
        "active":  "comparison",
    })

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse(request=request, name="chat.html", context={
        "active":   "chat",
        "datasets": db.list_datasets(),
    })


# ══════════════════════════════════════════════════════════════════════════════
# DATA API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/summary")
async def api_summary():
    return load_summary() or {}

@app.get("/api/traces")
async def api_traces(platform: str = None):
    return db.list_traces(platform=platform, limit=100)

@app.get("/api/traces/{trace_id}")
async def api_get_trace(trace_id: str):
    trace = db.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {**trace, "feedback": db.get_feedback(trace_id)}

@app.get("/api/experiments")
async def api_experiments():
    return [
        {**exp, "scores": db.get_aggregate_scores(exp["id"])}
        for exp in db.list_experiments()
    ]

@app.get("/api/datasets")
async def api_datasets():
    result = []
    for d in db.list_datasets():
        examples = db.list_examples(d["id"])
        result.append({**d, "example_count": len(examples)})
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CHAT API
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """
    Send a message to the real LangChain ReAct agent.

    Response: { response, trace_id, latency_ms, steps, status }
    """
    try:
        agent  = get_chat_agent()
        result = agent.chat(
            message=req.message,
            user_id=req.user_id,
            session_id=req.session_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# EVALOPS API
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/traces/{trace_id}/feedback")
async def api_trace_feedback(trace_id: str, req: FeedbackRequest):
    """Attach a user feedback score (thumbs up/down) to a trace."""
    trace = db.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    db.save_feedback(trace_id, req.key, req.score, req.comment)
    return {"success": True, "trace_id": trace_id, "key": req.key, "score": req.score}


@app.post("/api/traces/{trace_id}/add-to-dataset")
async def api_add_trace_to_dataset(trace_id: str, req: AddToDatasetRequest):
    """
    Promote a production trace into a dataset example (core EvalOps workflow).

    Creates the named dataset if it doesn't exist yet.
    """
    dataset = db.get_or_create_dataset(
        name=req.dataset_name,
        description="Production chat traces promoted to eval dataset",
    )
    ex_id = db.add_trace_to_dataset(trace_id, dataset["id"])
    if ex_id is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    examples = db.list_examples(dataset["id"])
    return {
        "success":        True,
        "example_id":     ex_id,
        "dataset_id":     dataset["id"],
        "dataset_name":   req.dataset_name,
        "total_examples": len(examples),
        "message": (f"Trace added to '{req.dataset_name}' "
                    f"({len(examples)} examples total)"),
    }


@app.post("/api/datasets/{dataset_id}/run-eval")
async def api_run_eval(dataset_id: str, evaluators: list[str] = None):
    """
    Run LLM-as-a-judge evaluators over every example in the dataset.

    evaluators: list of evaluator names (default: all three).
    Returns: experiment_id + aggregate scores.
    """
    all_datasets = db.list_datasets()
    dataset = next((d for d in all_datasets if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    examples = db.list_examples(dataset_id)
    if not examples:
        raise HTTPException(status_code=400, detail="Dataset has no examples")

    active_evaluators = {
        k: v for k, v in EVALUATORS.items()
        if evaluators is None or k in evaluators
    }

    exp_name = f"eval-{dataset['name']}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    exp_id   = db.create_experiment(
        name=exp_name,
        dataset_id=dataset_id,
        agent_version="v1-live",
        platform="langchain",
        metadata={"triggered_by": "dashboard_ui",
                  "evaluators": list(active_evaluators.keys())},
    )

    results, errors = [], []
    for ex in examples:
        question  = ex["inputs"].get("question", "")
        reference = ex["outputs"].get("answer", "")
        try:
            agent  = get_chat_agent()
            answer = agent.chat(question, user_id="eval_runner")["response"]
        except Exception as e:
            errors.append({"example_id": ex["id"], "error": str(e)})
            continue

        for ev_name, ev_fn in active_evaluators.items():
            try:
                r = ev_fn(question, answer, reference)
                db.save_score(
                    experiment_id=exp_id,
                    example_id=ex["id"],
                    input_text=question,
                    output_text=answer,
                    evaluator=ev_name,
                    score=r.get("score", 0.0),
                    comment=r.get("comment", r.get("reason", "")),
                )
                results.append({"example_id": ex["id"], "evaluator": ev_name,
                                 "score": r.get("score", 0.0)})
            except Exception as e:
                errors.append({"example_id": ex["id"], "evaluator": ev_name,
                                "error": str(e)})

    return {
        "success":            True,
        "experiment_id":      exp_id,
        "experiment_name":    exp_name,
        "examples_evaluated": len(examples),
        "scores_saved":       len(results),
        "errors":             errors,
        "aggregate_scores":   db.get_aggregate_scores(exp_id),
    }
