"""FastAPI dashboard — EvalOps POC comparison UI."""

import os, sys, json, uuid
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from phase3_poc.bridge.db import BridgeDB
from phase3_poc.bridge.evaluator import EVALUATORS

app       = FastAPI(title="Lyzr EvalOps POC")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
db        = BridgeDB()

# ── Lazy-load the agent (avoids slow startup, loads on first /api/chat call) ──
_agent = None

def get_chat_agent():
    global _agent
    if _agent is None:
        from phase3_poc.agents.langchain_agent import get_agent
        _agent = get_agent(db=db, agent_version="v1-live")
    return _agent

# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    user_id: str = "anon"
    session_id: str | None = None

class AddToDatasetRequest(BaseModel):
    dataset_name: str = "production-chat-traces"

SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "..", "poc_summary.json")

def load_summary():
    try:
        with open(SUMMARY_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    summary = load_summary()
    return templates.TemplateResponse(request=request, name="index.html", context={
        "summary": summary,
    })

@app.get("/traces", response_class=HTMLResponse)
async def traces(request: Request, platform: str = None):
    lyzr_traces = db.list_traces(platform="lyzr",      limit=50)
    ls_traces   = db.list_traces(platform="langsmith",  limit=50)
    return templates.TemplateResponse(request=request, name="traces.html", context={
        "lyzr_traces": lyzr_traces,
        "ls_traces":   ls_traces,
        "active":      "traces",
    })

@app.get("/datasets", response_class=HTMLResponse)
async def datasets(request: Request):
    datasets  = db.list_datasets()
    examples  = []
    if datasets:
        examples = db.list_examples(datasets[0]["id"])
    return templates.TemplateResponse(request=request, name="datasets.html", context={
        "datasets": datasets,
        "examples": examples,
        "active":   "datasets",
    })

@app.get("/experiments", response_class=HTMLResponse)
async def experiments(request: Request):
    experiments = db.list_experiments()
    exp_data    = []
    for exp in experiments:
        scores = db.get_aggregate_scores(exp["id"])
        exp_data.append({**exp, "scores": scores})
    return templates.TemplateResponse(request=request, name="experiments.html", context={
        "experiments": exp_data,
        "active":      "experiments",
    })

@app.get("/comparison", response_class=HTMLResponse)
async def comparison(request: Request):
    summary = load_summary()
    return templates.TemplateResponse(request=request, name="comparison.html", context={
        "summary": summary,
        "active":  "comparison",
    })

# ── API endpoints (for dashboard JS) ──────────────────────────────────────────

@app.get("/api/summary")
async def api_summary():
    return load_summary() or {}

@app.get("/api/traces")
async def api_traces(platform: str = None):
    return db.list_traces(platform=platform, limit=100)

@app.get("/api/experiments")
async def api_experiments():
    result = []
    for exp in db.list_experiments():
        result.append({**exp, "scores": db.get_aggregate_scores(exp["id"])})
    return result


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """
    Send a message to the real LangChain agent.

    Flow:
      1. Message → LangChain ReAct agent (FAISS retrieval + ticket tools)
      2. Response + intermediate steps returned
      3. Trace automatically saved to BridgeDB by TracedAgent.chat()

    Response:
      {
        "response":   str,          the agent's final answer
        "trace_id":   str,          UUID of the saved trace
        "latency_ms": float,
        "steps":      list,         [{tool, tool_input, observation}, ...]
        "status":     "success"|"error"
      }
    """
    try:
        agent = get_chat_agent()
        result = agent.chat(
            message=req.message,
            user_id=req.user_id,
            session_id=req.session_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Trace management endpoints ────────────────────────────────────────────────

@app.get("/api/traces/{trace_id}")
async def api_get_trace(trace_id: str):
    """Return a single trace with its feedback."""
    trace = db.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    feedback = db.get_feedback(trace_id)
    return {**trace, "feedback": feedback}


@app.post("/api/traces/{trace_id}/add-to-dataset")
async def api_add_trace_to_dataset(trace_id: str, req: AddToDatasetRequest):
    """
    Promote a production trace into a dataset example.

    This is the core EvalOps feedback loop:
      Production trace → human review → click 'Add to Dataset'
      → labeled example is created → future evals use real queries

    Creates the dataset if it doesn't exist yet.
    """
    # Ensure the dataset exists
    dataset = db.get_or_create_dataset(
        name=req.dataset_name,
        description="Production chat traces promoted to eval dataset",
    )

    ex_id = db.add_trace_to_dataset(trace_id, dataset["id"])
    if ex_id is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    examples = db.list_examples(dataset["id"])
    return {
        "success":    True,
        "example_id": ex_id,
        "dataset_id": dataset["id"],
        "dataset_name": req.dataset_name,
        "total_examples": len(examples),
        "message": f"Trace added to dataset '{req.dataset_name}' ({len(examples)} examples total)",
    }


# ── Dataset evaluation endpoint ───────────────────────────────────────────────

@app.post("/api/datasets/{dataset_id}/run-eval")
async def api_run_eval(dataset_id: str):
    """
    Run all 3 evaluators (correctness, helpfulness, conciseness) over every
    example in the dataset and save scores to a new experiment.

    Returns the experiment_id and aggregate scores so the UI can display results.
    """
    # Verify dataset exists
    datasets = db.list_datasets()
    dataset = next((d for d in datasets if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    examples = db.list_examples(dataset_id)
    if not examples:
        raise HTTPException(status_code=400, detail="Dataset has no examples to evaluate")

    # Create a new experiment record
    exp_name = f"eval-{dataset['name']}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    exp_id   = db.create_experiment(
        name=exp_name,
        dataset_id=dataset_id,
        agent_version="v1-live",
        platform="langchain",
        metadata={"triggered_by": "dashboard_ui", "evaluators": list(EVALUATORS.keys())},
    )

    # Run each evaluator over each example
    results = []
    errors  = []
    for ex in examples:
        question  = ex["inputs"].get("question", "")
        reference = ex["outputs"].get("answer", "")

        # Get the current agent's answer for this question
        try:
            agent  = get_chat_agent()
            result = agent.chat(question, user_id="eval_runner")
            answer = result["response"]
        except Exception as e:
            errors.append({"example_id": ex["id"], "error": str(e)})
            continue

        # Score with each evaluator
        for ev_name, ev_fn in EVALUATORS.items():
            try:
                score_result = ev_fn(question, answer, reference)
                score   = score_result.get("score", 0.0)
                comment = score_result.get("reason", "")
                db.save_score(
                    experiment_id=exp_id,
                    example_id=ex["id"],
                    input_text=question,
                    output_text=answer,
                    evaluator=ev_name,
                    score=score,
                    comment=comment,
                )
                results.append({"example_id": ex["id"], "evaluator": ev_name, "score": score})
            except Exception as e:
                errors.append({"example_id": ex["id"], "evaluator": ev_name, "error": str(e)})

    agg = db.get_aggregate_scores(exp_id)

    return {
        "success":       True,
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "examples_evaluated": len(examples),
        "scores_saved":  len(results),
        "errors":        errors,
        "aggregate_scores": agg,
    }


# ── Chat page route ───────────────────────────────────────────────────────────

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Live chat UI — messages go to the real LangChain agent."""
    datasets = db.list_datasets()
    return templates.TemplateResponse(request=request, name="chat.html", context={
        "active":   "chat",
        "datasets": datasets,
    })
