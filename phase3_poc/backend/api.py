"""
api.py — FastAPI application.

Page routes  → /  /traces  /datasets  /evaluations  /prompts  /chat
Data API     → /api/traces  /api/datasets  /api/evaluations  /api/evaluator-definitions
Chat API     → POST /api/chat/lyzr
EvalOps API  → POST /api/traces/bulk-add-to-dataset
               POST /api/traces/{id}/feedback
               POST /api/datasets/{id}/run-eval
               POST /api/evaluator-definitions
               POST /api/prompts  /api/prompts/{id}/activate  /api/prompts/suggest
"""

import os, sys, json, uuid
from datetime import datetime
from typing import Optional, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from phase3_poc.backend.db import BridgeDB
from phase3_poc.backend.evaluator import EVALUATORS, run_all_models, get_judge_prompt, EVAL_MODELS

app = FastAPI(title="Lyzr EvalOps")

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "ui", "templates")
templates     = Jinja2Templates(directory=TEMPLATES_DIR)
db            = BridgeDB()
SUMMARY_PATH  = os.path.join(os.path.dirname(__file__), "..", "poc_summary.json")

# ── Lazy agent loaders ────────────────────────────────────────────────────────
_lyzr_agents: dict = {}

def get_lyzr_agent():
    active = db.get_active_prompt()
    if not active:
        db.seed_default_prompts()
        active = db.get_active_prompt()
    prompt_id = active["id"]
    if prompt_id not in _lyzr_agents:
        from phase3_poc.agents.lyzr_adk_agent.agent import build_traced_agent_with_prompt
        _lyzr_agents[prompt_id] = build_traced_agent_with_prompt(
            system_prompt=active["content"],
            version=active["version"],
            db=db,
        )
    return _lyzr_agents[prompt_id], active


# ── Request models ────────────────────────────────────────────────────────────
class LyzrChatRequest(BaseModel):
    message:    str
    user_id:    str = "anon"
    session_id: Optional[str] = None

class BulkAddToDatasetRequest(BaseModel):
    trace_ids:    List[str]
    dataset_name: str

class FeedbackRequest(BaseModel):
    score:   float
    comment: str = ""
    key:     str = "user_feedback"

class CreatePromptRequest(BaseModel):
    name:       str
    content:    str
    notes:      str = ""
    set_active: bool = False

class RunEvalRequest(BaseModel):
    evaluator_ids: List[str] = []   # empty = use all defined evaluators
    eval_name:     str = ""
    models:        List[str] = []   # empty = use all EVAL_MODELS

class CreateEvaluatorRequest(BaseModel):
    name:         str
    description:  str
    judge_prompt: str


# ── Helpers ───────────────────────────────────────────────────────────────────
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
async def traces_page(request: Request):
    return templates.TemplateResponse(request=request, name="traces.html", context={
        "traces":   db.list_traces(platform="lyzr", limit=200),
        "datasets": db.list_datasets(),
        "active":   "traces",
    })

@app.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    all_datasets = db.list_datasets()
    datasets_with_counts = []
    for d in all_datasets:
        examples = db.list_examples(d["id"])
        datasets_with_counts.append({**d, "examples": examples, "example_count": len(examples)})
    return templates.TemplateResponse(request=request, name="datasets.html", context={
        "datasets": datasets_with_counts,
        "active":   "datasets",
    })

@app.get("/evaluations", response_class=HTMLResponse)
async def evaluations_page(request: Request):
    evals = db.list_evaluations()
    evals_enriched = []
    for ev in evals:
        scores = db.get_aggregate_scores(ev["id"])
        meta   = json.loads(ev.get("metadata_json") or "{}")
        evals_enriched.append({**ev, "scores": scores, "meta": meta})
    evaluator_defs = db.list_evaluator_definitions()
    datasets       = db.list_datasets()
    return templates.TemplateResponse(request=request, name="evaluations.html", context={
        "evaluations":     evals_enriched,
        "evaluator_defs":  evaluator_defs,
        "datasets":        datasets,
        "eval_models":     EVAL_MODELS,
        "active":          "evaluations",
    })

@app.get("/evaluations/{eval_id}", response_class=HTMLResponse)
async def evaluation_detail_page(request: Request, eval_id: str):
    ev = db.get_evaluation(eval_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    meta    = json.loads(ev.get("metadata_json") or "{}")
    models  = meta.get("models", EVAL_MODELS)
    ev_names = meta.get("evaluators", [])
    # Build detail rows per example
    scores_by_model = db.get_scores_by_model(eval_id)
    # Collect all example ids in order
    dataset = db.list_examples(ev["dataset_id"]) if ev.get("dataset_id") else []
    return templates.TemplateResponse(request=request, name="evaluation_detail.html", context={
        "evaluation":      ev,
        "models":          models,
        "evaluator_names": ev_names,
        "scores_by_model": scores_by_model,
        "examples":        dataset,
        "active":          "evaluations",
    })

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    active_prompt = db.get_active_prompt()
    return templates.TemplateResponse(request=request, name="chat.html", context={
        "active":        "chat",
        "datasets":      db.list_datasets(),
        "active_prompt": active_prompt,
    })

@app.get("/prompts", response_class=HTMLResponse)
async def prompts_page(request: Request):
    prompts    = db.list_prompts()
    active     = db.get_active_prompt()
    evals      = db.list_evaluations()
    ev_scores  = {e["id"]: db.get_aggregate_scores(e["id"]) for e in evals}
    return templates.TemplateResponse(request=request, name="prompts.html", context={
        "active":        "prompts",
        "prompts":       prompts,
        "active_prompt": active,
        "evaluations":   evals,
        "ev_scores":     ev_scores,
    })


# ══════════════════════════════════════════════════════════════════════════════
# DATA API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/traces")
async def api_traces(platform: str = "lyzr", limit: int = 100):
    return db.list_traces(platform=platform, limit=limit)

@app.get("/api/traces/{trace_id}")
async def api_get_trace(trace_id: str):
    trace = db.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {**trace, "feedback": db.get_feedback(trace_id)}

@app.get("/api/datasets")
async def api_datasets():
    result = []
    for d in db.list_datasets():
        examples = db.list_examples(d["id"])
        result.append({**d, "example_count": len(examples)})
    return result

@app.get("/api/datasets/{dataset_id}/examples")
async def api_dataset_examples(dataset_id: str):
    examples = db.list_examples(dataset_id)
    return examples

@app.get("/api/evaluations")
async def api_evaluations():
    evals = db.list_evaluations()
    return [
        {**ev, "scores": db.get_aggregate_scores(ev["id"]),
         "meta": json.loads(ev.get("metadata_json") or "{}")}
        for ev in evals
    ]

@app.get("/api/evaluations/{eval_id}/scores")
async def api_evaluation_scores(eval_id: str):
    return db.get_scores_by_model(eval_id)

@app.get("/api/evaluator-definitions")
async def api_list_evaluator_defs():
    return db.list_evaluator_definitions()

@app.get("/api/summary")
async def api_summary():
    return load_summary() or {}


# ══════════════════════════════════════════════════════════════════════════════
# CHAT API
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat/lyzr")
async def api_chat_lyzr(req: LyzrChatRequest):
    try:
        tracer, active_prompt = get_lyzr_agent()
        output, trace_id = tracer.run(
            message=req.message,
            user_id=req.user_id,
            session_id=req.session_id or req.user_id,
        )
        trace = db.get_trace(trace_id) or {}
        return {
            "response":       output,
            "trace_id":       trace_id,
            "latency_ms":     trace.get("latency_ms"),
            "prompt_version": active_prompt["version"],
            "prompt_name":    active_prompt["name"],
            "status":         trace.get("status", "success"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# EVALOPS API
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/traces/bulk-add-to-dataset")
async def api_bulk_add(req: BulkAddToDatasetRequest):
    """Add multiple traces to a dataset in one call."""
    dataset = db.get_or_create_dataset(
        name=req.dataset_name,
        description="Traces promoted from dashboard",
    )
    added, skipped, failed = [], [], []
    for tid in req.trace_ids:
        ex_id = db.add_trace_to_dataset(tid, dataset["id"])
        if ex_id is None:
            failed.append(tid)
        else:
            added.append({"trace_id": tid, "example_id": ex_id})
    return {
        "success":      True,
        "dataset_id":   dataset["id"],
        "dataset_name": req.dataset_name,
        "added":        len(added),
        "failed":       len(failed),
        "total_examples": len(db.list_examples(dataset["id"])),
    }

@app.post("/api/traces/{trace_id}/feedback")
async def api_trace_feedback(trace_id: str, req: FeedbackRequest):
    trace = db.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    db.save_feedback(trace_id, req.key, req.score, req.comment)
    return {"success": True, "trace_id": trace_id, "key": req.key, "score": req.score}


@app.post("/api/datasets/{dataset_id}/run-eval")
async def api_run_eval(dataset_id: str, req: RunEvalRequest):
    """
    Run selected evaluators over every example in the dataset,
    across all selected models.
    """
    all_datasets = db.list_datasets()
    dataset = next((d for d in all_datasets if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    examples = db.list_examples(dataset_id)
    if not examples:
        raise HTTPException(status_code=400, detail="Dataset has no examples")

    # Resolve evaluators
    all_defs    = db.list_evaluator_definitions()
    if req.evaluator_ids:
        active_defs = [d for d in all_defs if d["id"] in req.evaluator_ids]
    else:
        active_defs = all_defs

    if not active_defs:
        raise HTTPException(status_code=400, detail="No evaluators selected")

    models     = req.models or EVAL_MODELS
    eval_name  = req.eval_name or f"eval-{dataset['name']}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    eval_id    = db.create_evaluation(
        name=eval_name,
        dataset_id=dataset_id,
        agent_version="v1-live",
        platform="lyzr",
        metadata={
            "triggered_by": "dashboard_ui",
            "evaluators":   [d["name"] for d in active_defs],
            "models":       models,
        },
    )

    results, errors = [], []
    tracer, _ = get_lyzr_agent()

    for ex in examples:
        question  = ex["inputs"].get("question", ex["inputs"].get("input", ""))
        reference = ex["outputs"].get("answer",   ex["outputs"].get("output", ""))

        try:
            answer, _ = tracer.run(question, user_id="eval_runner")
        except Exception as e:
            errors.append({"example_id": ex["id"], "error": str(e)})
            continue

        for ev_def in active_defs:
            prompt_template = ev_def["judge_prompt"]
            multi = run_all_models(
                judge_prompt_template=prompt_template,
                evaluator_name=ev_def["name"],
                input_text=question,
                output_text=answer,
                reference=reference,
                models=models,
            )
            for r in multi:
                try:
                    db.save_score(
                        evaluation_id=eval_id,
                        example_id=ex["id"],
                        input_text=question,
                        output_text=answer,
                        evaluator=r["evaluator"],
                        model=r["model"],
                        score=r["score"],
                        comment=r["comment"],
                    )
                    results.append(r)
                except Exception as e:
                    errors.append({"example_id": ex["id"], "evaluator": r["evaluator"],
                                   "model": r["model"], "error": str(e)})

    return {
        "success":            True,
        "evaluation_id":      eval_id,
        "evaluation_name":    eval_name,
        "examples_evaluated": len(examples),
        "scores_saved":       len(results),
        "errors":             errors,
        "aggregate_scores":   db.get_aggregate_scores(eval_id),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATOR DEFINITIONS API
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/evaluator-definitions")
async def api_create_evaluator_def(req: CreateEvaluatorRequest):
    ev_id = db.create_evaluator_definition(
        name=req.name,
        description=req.description,
        judge_prompt=req.judge_prompt,
    )
    return {"id": ev_id, "name": req.name}


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT MANAGEMENT API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/prompts")
async def api_list_prompts():
    return db.list_prompts()

@app.get("/api/prompts/active")
async def api_active_prompt():
    p = db.get_active_prompt()
    if not p:
        raise HTTPException(status_code=404, detail="No active prompt found")
    return p

@app.post("/api/prompts")
async def api_create_prompt(req: CreatePromptRequest):
    prompts  = db.list_prompts()
    version  = f"v{len(prompts) + 1}"
    prompt_id = db.create_prompt(
        version=version,
        name=req.name,
        content=req.content,
        notes=req.notes,
        set_active=req.set_active,
    )
    if req.set_active:
        _lyzr_agents.clear()
    return {"id": prompt_id, "version": version, "name": req.name, "active": req.set_active}

@app.post("/api/prompts/{prompt_id}/activate")
async def api_activate_prompt(prompt_id: str):
    ok = db.activate_prompt(prompt_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Prompt not found")
    _lyzr_agents.clear()
    prompt = db.get_prompt(prompt_id)
    return {"success": True, "active_version": prompt["version"], "name": prompt["name"]}

@app.post("/api/prompts/suggest")
async def api_suggest_prompt():
    from langchain_openai import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage

    active = db.get_active_prompt()
    if not active:
        raise HTTPException(status_code=400, detail="No active prompt to improve")

    all_feedback = []
    for t in db.list_traces(platform="lyzr", limit=50):
        fb = [f for f in db.get_feedback(t["id"]) if f["score"] == 0.0]
        if fb:
            all_feedback.append({"input": t["input"], "output": t["output"],
                                  "comment": fb[0].get("comment", "")})

    if not all_feedback:
        for ev in db.list_evaluations()[:3]:
            for s in db.get_scores(ev["id"]):
                if s["score"] < 0.5:
                    all_feedback.append({"input": s["input_text"],
                                         "output": s["output_text"],
                                         "comment": s.get("comment", "")})
            if len(all_feedback) >= 5:
                break

    if not all_feedback:
        all_feedback = [{"input": "(no failing traces found)", "output": "", "comment": ""}]

    examples_text = "\n\n".join(
        f"Q: {ex['input']}\nA: {ex['output']}"
        + (f"\nFeedback: {ex['comment']}" if ex['comment'] else "")
        for ex in all_feedback[:6]
    )

    llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert AI prompt engineer. Analyze failing agent responses and their "
            "evaluator feedback, then rewrite the system prompt to fix every observed issue.\n\n"
            "Pay close attention to feedback about:\n"
            "- Persona/voice (e.g. using 'we/our' vs third-person)\n"
            "- Tone and language (conversational vs formal)\n"
            "- Directness and friction reduction\n"
            "- Accuracy and intent understanding\n\n"
            "If persona failures are present, add an explicit mandatory persona rule that also "
            "covers rephrasing of any quoted documentation or context (not just direct statements)."
        )),
        HumanMessage(content=(
            f"CURRENT SYSTEM PROMPT:\n{active['content']}\n\n"
            f"FAILING EXAMPLES (with evaluator feedback):\n{examples_text}\n\n"
            "Write an improved system prompt that fixes ALL failure patterns above. "
            "Respond with ONLY the improved system prompt text, no explanation."
        ))
    ])

    return {
        "suggested_content":   response.content.strip(),
        "based_on_examples":   len(all_feedback),
        "current_version":     active["version"],
        "current_prompt_name": active["name"],
    }
