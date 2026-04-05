"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Phase 3 — POC Runner
 Generates ALL data needed for the comparison dashboard.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Steps:
  1. Run LangSmith agent on 7 queries   → traces in LangSmith cloud
  2. Run Lyzr ADK v1 agent on 7 queries → traces in local SQLite
  3. Run Lyzr ADK v2 agent on 7 queries → traces in local SQLite
  4. Build eval dataset from references
  5. Run LLM-as-a-judge evaluations:
       - LangSmith experiment (uses LangSmith evaluate())
       - Lyzr v1 experiment   (uses backend ExperimentRunner)
       - Lyzr v2 experiment   (uses backend ExperimentRunner)
  6. Store comparison results for the dashboard

HOW TO RUN:
  python phase3_poc/run_poc.py
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel   import Panel
from rich.rule    import Rule
from rich.table   import Table

# Backend layer
from phase3_poc.backend.db         import BridgeDB
from phase3_poc.backend.dataset    import DatasetStore
from phase3_poc.backend.experiment import ExperimentRunner
from phase3_poc.backend.evaluator  import correctness, helpfulness, conciseness

# Agents
from phase3_poc.agents.lyzr_adk_agent.agent import TEST_QUERIES

console = Console()
db      = BridgeDB()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — LangSmith agent
# ─────────────────────────────────────────────────────────────────────────────

def step1_langsmith():
    console.print(Rule("[bold blue]Step 1 — LangSmith Agent[/bold blue]"))
    from phase3_poc.agents.langsmith_agent import run_all, build_langsmith_agent
    from langsmith import Client
    from langsmith.evaluation import evaluate
    from langsmith.schemas import Run, Example
    from langchain.schema import HumanMessage
    from langchain_openai import ChatOpenAI

    console.print("[dim]Running 7 queries → LangSmith cloud traces...[/dim]")
    ls_results = run_all()
    console.print(f"[green]✓ {len(ls_results)} LangSmith traces captured[/green]")

    # Mirror LangSmith results into local DB for dashboard parity
    for r in ls_results:
        import uuid
        db.save_trace({
            "id":            str(uuid.uuid4()),
            "project":       os.getenv("LANGCHAIN_PROJECT", "lyzr-evalops-phase1"),
            "platform":      "langsmith",
            "input":         r["input"],
            "output":        r["output"],
            "latency_ms":    r["latency_ms"],
            "token_count":   len(r["input"] + r["output"]) // 4,
            "status":        "success",
            "user_id":       r["user_id"],
            "session_id":    f"ls-poc-{r['user_id']}",
            "agent_version": "v1",
            "timestamp":     __import__("datetime").datetime.utcnow().isoformat(),
        })

    console.print("[dim]Running LangSmith LLM-as-judge evaluations...[/dim]")

    def ls_runner(inputs):
        executor = build_langsmith_agent()
        return {"output": executor.invoke({"input": inputs["input"]})["output"]}

    def ls_correctness(run: Run, example: Example):
        judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        ans   = run.outputs.get("output","") if run.outputs else ""
        ref   = example.outputs.get("output","") if example.outputs else ""
        q     = example.inputs.get("input","")
        resp  = judge.invoke([HumanMessage(content=
            f"Question: {q}\nReference: {ref}\nAgent: {ans}\n"
            "SCORE: [0 or 1]\nREASON: [one sentence]")])
        score = 0.0
        for line in resp.content.split("\n"):
            if line.startswith("SCORE:"):
                try: score = float(line.replace("SCORE:","").strip())
                except: pass
        reason = next((l.replace("REASON:","").strip() for l in resp.content.split("\n")
                       if l.startswith("REASON:")), "")
        return {"key":"correctness","score":score,"comment":reason}

    def ls_helpfulness(run: Run, example: Example):
        judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        ans   = run.outputs.get("output","") if run.outputs else ""
        q     = example.inputs.get("input","")
        resp  = judge.invoke([HumanMessage(content=
            f"Question: {q}\nAgent: {ans}\n"
            "Rate helpfulness 0.0-1.0.\nSCORE: [0.0-1.0]\nREASON: [one sentence]")])
        score = 0.5
        for line in resp.content.split("\n"):
            if line.startswith("SCORE:"):
                try: score = float(line.replace("SCORE:","").strip())
                except: pass
        reason = next((l.replace("REASON:","").strip() for l in resp.content.split("\n")
                       if l.startswith("REASON:")), "")
        return {"key":"helpfulness","score":score,"comment":reason}

    def ls_conciseness(run: Run, example: Example):
        judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        ans   = run.outputs.get("output","") if run.outputs else ""
        resp  = judge.invoke([HumanMessage(content=
            f"Response: {ans}\nRate conciseness 0.0-1.0.\nSCORE: [0.0-1.0]\nREASON: [one sentence]")])
        score = 0.5
        for line in resp.content.split("\n"):
            if line.startswith("SCORE:"):
                try: score = float(line.replace("SCORE:","").strip())
                except: pass
        reason = next((l.replace("REASON:","").strip() for l in resp.content.split("\n")
                       if l.startswith("REASON:")), "")
        return {"key":"conciseness","score":score,"comment":reason}

    ls_eval_results = evaluate(
        ls_runner,
        data="tech-support-agent-v1",
        evaluators=[ls_correctness, ls_helpfulness, ls_conciseness],
        experiment_prefix="poc-langsmith-v1",
        max_concurrency=2,
    )

    ls_scores = {"correctness": [], "helpfulness": [], "conciseness": []}
    try:
        df = ls_eval_results.to_pandas()
        for key in ls_scores:
            col = f"feedback.{key}"
            if col in df.columns:
                ls_scores[key] = df[col].dropna().tolist()
    except Exception:
        pass

    if all(len(v) == 0 for v in ls_scores.values()):
        for result in ls_eval_results:
            fb = getattr(result, "feedback_results", []) or []
            for f in fb:
                k = getattr(f, "key", None)
                s = getattr(f, "score", None)
                if k in ls_scores and s is not None:
                    ls_scores[k].append(float(s))

    avg_scores = {k: round(sum(v)/len(v),3) if v else 0.0 for k,v in ls_scores.items()}
    console.print(f"[green]✓ LangSmith eval scores: {avg_scores}[/green]")

    ls_exp_id = db.create_experiment(
        name="poc-langsmith-v1",
        dataset_id=(db.get_dataset_by_name("tech-support-v1") or {}).get("id", "n/a"),
        agent_version="v1",
        platform="langsmith",
        metadata=avg_scores,
    )
    examples = db.list_examples(
        (db.get_dataset_by_name("tech-support-v1") or {}).get("id", "")
    ) if db.get_dataset_by_name("tech-support-v1") else []
    for ex in examples:
        for ev, avg in avg_scores.items():
            db.save_score(ls_exp_id, ex["id"], ex["inputs"].get("input",""),
                          "", ev, avg, "LangSmith aggregate score")

    return ls_exp_id, avg_scores


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2+3 — Lyzr ADK v1 + v2 agents
# ─────────────────────────────────────────────────────────────────────────────

def step2_lyzr_agents():
    console.print(Rule("[bold magenta]Steps 2+3 — Lyzr ADK v1 + v2 Agents[/bold magenta]"))
    from phase3_poc.agents.lyzr_adk_agent.agent import run_all

    console.print("[dim]Running Lyzr ADK v1 (7 queries)...[/dim]")
    results_v1, tracer_v1 = run_all(version="v1", db=db)
    console.print(f"[green]✓ Lyzr v1: {len(results_v1)} traces captured[/green]")

    console.print("[dim]Running Lyzr ADK v2 (7 queries)...[/dim]")
    results_v2, tracer_v2 = run_all(version="v2", db=db)
    console.print(f"[green]✓ Lyzr v2: {len(results_v2)} traces captured[/green]")

    return results_v1, results_v2, tracer_v1, tracer_v2


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

def step4_dataset():
    console.print(Rule("[bold cyan]Step 4 — Build Evaluation Dataset[/bold cyan]"))
    store      = DatasetStore(db=db)
    dataset_id = store.create_from_reference("tech-support-v1")
    examples   = db.list_examples(dataset_id)
    console.print(f"[green]✓ Dataset 'tech-support-v1' — {len(examples)} examples[/green]")
    return dataset_id


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Evaluations
# ─────────────────────────────────────────────────────────────────────────────

def step5_evaluations():
    console.print(Rule("[bold yellow]Step 5 — LLM-as-a-Judge Evaluations[/bold yellow]"))
    from phase3_poc.agents.lyzr_adk_agent.agent import build_traced_agent
    from phase3_poc.agents.langsmith_agent import build_langsmith_agent

    runner = ExperimentRunner(db=db)

    def lyzr_v1_runner(input_text: str) -> str:
        tracer = build_traced_agent(version="v1", db=db)
        output, _ = tracer.run(input_text, user_id="eval-runner")
        return output

    def lyzr_v2_runner(input_text: str) -> str:
        tracer = build_traced_agent(version="v2", db=db)
        output, _ = tracer.run(input_text, user_id="eval-runner")
        return output

    console.print("[dim]Evaluating Lyzr ADK v1...[/dim]")
    exp_lyzr_v1 = runner.run(lyzr_v1_runner, "tech-support-v1",
                              experiment_name="poc-lyzr-v1",
                              agent_version="v1", platform="lyzr")

    console.print("[dim]Evaluating Lyzr ADK v2...[/dim]")
    exp_lyzr_v2 = runner.run(lyzr_v2_runner, "tech-support-v1",
                              experiment_name="poc-lyzr-v2",
                              agent_version="v2", platform="lyzr")

    return exp_lyzr_v1, exp_lyzr_v2


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Comparison summary
# ─────────────────────────────────────────────────────────────────────────────

def step6_comparison(ls_exp_id, ls_scores, exp_lyzr_v1, exp_lyzr_v2):
    console.print(Rule("[bold green]Step 6 — Comparison Summary[/bold green]"))
    runner = ExperimentRunner(db=db)

    lyzr_v1_scores = db.get_aggregate_scores(exp_lyzr_v1)
    lyzr_v2_scores = db.get_aggregate_scores(exp_lyzr_v2)
    decision       = runner.compare(exp_lyzr_v1, exp_lyzr_v2)

    table = Table(title="LangSmith vs Lyzr+Bridge — Evaluation Scores", show_lines=True)
    table.add_column("Metric",       style="bold")
    table.add_column("LangSmith v1", justify="center")
    table.add_column("Lyzr ADK v1",  justify="center")
    table.add_column("Lyzr ADK v2",  justify="center")
    table.add_column("Δ v1→v2",      justify="center")

    for metric in ["correctness", "helpfulness", "conciseness"]:
        ls  = ls_scores.get(metric, 0.0)
        l1  = lyzr_v1_scores.get(metric, 0.0)
        l2  = lyzr_v2_scores.get(metric, 0.0)
        d   = round(l2 - l1, 3)
        col = "green" if d > 0.03 else ("red" if d < -0.03 else "yellow")
        table.add_row(metric, str(ls), str(l1), str(l2), f"[{col}]{d:+.3f}[/{col}]")

    console.print(table)

    d_color = {"PROMOTE":"green","CANARY":"yellow","HOLD":"yellow","ROLLBACK":"red"}
    color   = d_color.get(decision["verdict"], "white")
    console.print(Panel(
        f"[bold {color}]Release Decision: {decision['verdict']}[/bold {color}]\n\n"
        f"{decision['reason']}",
        title="[bold]EvalOps Decision Gate[/bold]",
        border_style=color,
    ))

    summary = {
        "langsmith": {"experiment_id": ls_exp_id, "scores": ls_scores},
        "lyzr_v1":   {"experiment_id": exp_lyzr_v1, "scores": lyzr_v1_scores},
        "lyzr_v2":   {"experiment_id": exp_lyzr_v2, "scores": lyzr_v2_scores},
        "decision":  decision,
    }
    summary_path = os.path.join(os.path.dirname(__file__), "poc_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[dim]Summary saved to {summary_path}[/dim]")
    console.print("\n[bold green]✓ All data generated. Launch dashboard:[/bold green]")
    console.print("  uvicorn phase3_poc.backend.api:app --reload --port 8000\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Phase 3 — EvalOps POC Runner[/bold cyan]\n"
        "LangSmith vs Lyzr+Bridge — Full Comparison\n\n"
        "[dim]Generates all traces, datasets, and evaluation scores\n"
        "needed for the comparison dashboard.[/dim]",
        border_style="cyan"
    ))

    step4_dataset()
    ls_exp_id, ls_scores = step1_langsmith()
    step2_lyzr_agents()
    exp_lyzr_v1, exp_lyzr_v2 = step5_evaluations()
    step6_comparison(ls_exp_id, ls_scores, exp_lyzr_v1, exp_lyzr_v2)
