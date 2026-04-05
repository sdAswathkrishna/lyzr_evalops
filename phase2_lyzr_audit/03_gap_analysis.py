"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE 2 — STEP 3: Full Gap Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THIS DOES:
  Produces the definitive side-by-side comparison of:
    LangSmith (what you learned in Phase 1)
    vs
    Lyzr (what you just audited in Phase 2)

  Then outputs a prioritised Phase 3 build list — what
  to build, in what order, and why.

HOW TO RUN:
  python phase2_lyzr_audit/03_gap_analysis.py
"""

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  THE FULL FEATURE MATRIX
#  Evidence column = where we found (or didn't find) the feature
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_MATRIX = [
    # (Feature, LangSmith status, Lyzr status, Gap size, Evidence / note)
    (
        "Automatic tracing of agent runs",
        "✅ Full",
        "✅ Full",
        "None",
        "Lyzr has OTel-backed traces in Studio. Both auto-trace."
    ),
    (
        "Per-span details\n(prompt text, LLM response body)",
        "✅ Full",
        "⚠️  Partial",
        "Small",
        "Lyzr shows duration + token counts but raw prompt/response\n"
        "body not confirmed in docs."
    ),
    (
        "Token counts per span",
        "✅ Full",
        "✅ Full",
        "None",
        "Lyzr dashboard shows input + output tokens, avg per trace."
    ),
    (
        "Latency per span",
        "✅ Full",
        "✅ Full",
        "None",
        "Lyzr Operation Waterfall shows per-span duration."
    ),
    (
        "Programmatic trace access\n(list/get runs via SDK)",
        "✅ Full",
        "❌ Missing",
        "CRITICAL",
        "LangSmith: client.list_runs(). "
        "Lyzr: no list_traces() or get_run() on Studio or AgentAPI."
    ),
    (
        "Trace metadata & tags\n(user_id, version, env)",
        "✅ Full",
        "⚠️  Partial",
        "Medium",
        "Lyzr captures user_id, session_id, agent_id. "
        "No free-form tag dict like LangSmith's metadata={}."
    ),
    (
        "Feedback API\n(attach score to a run)",
        "✅ Full",
        "❌ Missing",
        "CRITICAL",
        "LangSmith: client.create_feedback(run_id, key, score). "
        "Lyzr: thumbs up/down in UI only. No SDK method."
    ),
    (
        "Multi-dimensional feedback\n(correctness=0.9, helpfulness=0.7)",
        "✅ Full",
        "❌ Missing",
        "CRITICAL",
        "LangSmith supports arbitrary key-score pairs per run. "
        "Lyzr: binary only (thumbs up/down)."
    ),
    (
        "Evaluation dataset management\n(create, add, version datasets)",
        "✅ Full",
        "❌ Missing",
        "CRITICAL",
        "LangSmith: client.create_dataset(), client.create_example(). "
        "Lyzr: no dataset store. A-Sim uses personas/scenarios instead."
    ),
    (
        "Traces → Dataset pipeline\n(save production run as eval example)",
        "✅ Full",
        "❌ Missing",
        "CRITICAL",
        "LangSmith: select a trace, add to dataset with one click or API. "
        "Lyzr: no such workflow exists."
    ),
    (
        "Built-in safety evaluators\n(toxicity, PII, hallucination)",
        "❌ Missing",
        "✅ Full",
        "Lyzr wins",
        "LangSmith has no built-in safety evals. "
        "Lyzr: 9-type PII, toxicity ML model, prompt injection, bias."
    ),
    (
        "Custom LLM-as-a-judge evaluators\n(user-defined scoring functions)",
        "✅ Full",
        "❌ Missing",
        "CRITICAL",
        "LangSmith: evaluate(runner, evaluators=[my_fn]). "
        "Lyzr: no API to register a custom evaluator function."
    ),
    (
        "Criteria-based evaluation\n(score on rubric without reference)",
        "✅ Full",
        "❌ Missing",
        "HIGH",
        "LangSmith: criteria evaluator works with just question + answer. "
        "Lyzr: no equivalent."
    ),
    (
        "Reference-based evaluation\n(compare to ground truth answer)",
        "✅ Full",
        "❌ Missing",
        "HIGH",
        "LangSmith: correctness evaluator uses reference_output. "
        "Lyzr: no reference output concept in A-Sim."
    ),
    (
        "Experiment runner\n(agent × dataset → scores)",
        "✅ Full",
        "❌ Missing",
        "CRITICAL",
        "LangSmith: evaluate(runner, data=dataset). "
        "Lyzr: A-Sim runs scenarios but not against a flat dataset."
    ),
    (
        "Experiment comparison UI\n(v1 vs v2 on same dataset)",
        "✅ Full",
        "❌ Missing",
        "CRITICAL",
        "LangSmith: Compare Experiments tab with diff view. "
        "Lyzr: sequential hardening rounds, not parallel A/B."
    ),
    (
        "Release decision gate\n(PROMOTE / CANARY / HOLD / ROLLBACK)",
        "⚠️  Manual",
        "❌ Missing",
        "HIGH",
        "LangSmith: you build the logic yourself (as we did in Step 5). "
        "Lyzr: no concept. This is fully custom to build."
    ),
    (
        "Canary deployment\n(route X% traffic to new version)",
        "❌ Missing",
        "❌ Missing",
        "Both need it",
        "Neither platform has this. Pure infrastructure build needed."
    ),
    (
        "Agent version rollback",
        "❌ Missing",
        "⚠️  Partial",
        "Medium",
        "Lyzr mentions version management in Settings. "
        "No documented one-click rollback."
    ),
    (
        "Auto-generated test cases\n(persona × scenario matrix)",
        "❌ Missing",
        "✅ Full",
        "Lyzr wins",
        "Lyzr A-Sim: define personas + scenarios, AI generates test cases. "
        "LangSmith: you write every test case manually."
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
#  GAP SEVERITY COLOURS
# ─────────────────────────────────────────────────────────────────────────────

GAP_COLORS = {
    "CRITICAL":     "red",
    "HIGH":         "yellow",
    "Medium":       "blue",
    "Small":        "dim",
    "None":         "green",
    "Lyzr wins":    "magenta",
    "Both need it": "yellow",
}


def print_feature_matrix():
    console.print(Rule("[bold cyan]Full Feature Comparison: LangSmith vs Lyzr[/bold cyan]"))

    table = Table(
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
        title="LangSmith vs Lyzr — EvalOps Feature Audit",
    )
    table.add_column("Feature",         width=30, style="bold")
    table.add_column("LangSmith",       width=12, justify="center")
    table.add_column("Lyzr",            width=12, justify="center")
    table.add_column("Gap",             width=14, justify="center")
    table.add_column("Evidence",        width=45)

    for feature, ls_status, lyzr_status, gap, note in FEATURE_MATRIX:
        color = GAP_COLORS.get(gap, "white")
        table.add_row(
            feature,
            ls_status,
            lyzr_status,
            f"[{color}]{gap}[/{color}]",
            note,
        )

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
#  GAP COUNT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_gap_summary():
    console.print(Rule("[bold yellow]Gap Summary[/bold yellow]"))

    counts = {}
    for _, _, _, gap, _ in FEATURE_MATRIX:
        counts[gap] = counts.get(gap, 0) + 1

    summary = Table(border_style="yellow")
    summary.add_column("Gap Severity", style="bold")
    summary.add_column("Count", justify="center")
    summary.add_column("Meaning")

    severity_info = {
        "CRITICAL":     ("red",     "Lyzr is completely missing this — must build for Phase 3"),
        "HIGH":         ("yellow",  "Significant gap — important but not day-1 blocker"),
        "Medium":       ("blue",    "Partial support exists — enhancement needed"),
        "Small":        ("dim",     "Minor gap — likely easy to fill"),
        "None":         ("green",   "Lyzr and LangSmith are equivalent"),
        "Lyzr wins":    ("magenta", "Lyzr is BETTER than LangSmith here"),
        "Both need it": ("yellow",  "Neither platform has this — net-new infrastructure build"),
    }

    for gap, (color, meaning) in severity_info.items():
        count = counts.get(gap, 0)
        if count > 0:
            summary.add_row(f"[{color}]{gap}[/{color}]", str(count), meaning)

    console.print(summary)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 BUILD PLAN — prioritised
# ─────────────────────────────────────────────────────────────────────────────

PHASE3_PLAN = [
    {
        "priority": 1,
        "name": "Trace Access API",
        "what": "A service that reads Lyzr traces and exposes them as Python objects",
        "why": "Everything else depends on this — you can't build a dataset or eval pipeline without programmatic trace access",
        "how": "Wrap Lyzr's internal trace endpoints (OTel/REST) + expose list_runs(project, filters) and get_run(run_id)",
        "effort": "Medium",
        "maps_to": "LangSmith: client.list_runs() + client.get_run()",
    },
    {
        "priority": 2,
        "name": "Feedback API",
        "what": "REST endpoint + Python client to attach scores to a Lyzr run by ID",
        "why": "Without feedback, you can't measure what's good vs bad. It's the signal for all downstream EvalOps.",
        "how": "POST /feedback {run_id, key, score, comment} → stored in a DB alongside the trace",
        "effort": "Small",
        "maps_to": "LangSmith: client.create_feedback(run_id, key, score)",
    },
    {
        "priority": 3,
        "name": "Dataset Store",
        "what": "A store for (input, reference_output, metadata) eval examples with versioning",
        "why": "The 'answer key' for your agent. Without it, evaluations have no ground truth to compare against.",
        "how": "Postgres or SQLite table: datasets + examples. API: create_dataset, add_example, list_examples",
        "effort": "Medium",
        "maps_to": "LangSmith: client.create_dataset() + client.create_example()",
    },
    {
        "priority": 4,
        "name": "Trace → Dataset Pipeline",
        "what": "UI + API flow: select a production trace → curate it → add to dataset",
        "why": "Real users produce edge cases you never write manually. This closes the production → eval loop.",
        "how": "Lyzr Studio plugin: 'Save to Dataset' button on any trace row",
        "effort": "Medium",
        "maps_to": "LangSmith: 'Add to Dataset' button in Traces UI",
    },
    {
        "priority": 5,
        "name": "Custom Evaluator Registry",
        "what": "Let users define Python scoring functions and register them as named evaluators",
        "why": "Safety evals (Lyzr's strength) are not enough. Teams need correctness, helpfulness, task-completion — domain-specific criteria.",
        "how": "Decorator or class: @evaluator def correctness(run, example) → {score, comment}. Store + re-run on demand.",
        "effort": "Medium",
        "maps_to": "LangSmith: custom evaluator functions passed to evaluate()",
    },
    {
        "priority": 6,
        "name": "Experiment Runner",
        "what": "Run an agent against a dataset with one or more evaluators, store all scores",
        "why": "Turns the dataset + evaluators into a reproducible, numbered experiment you can compare across versions.",
        "how": "evaluate(agent_fn, dataset_name, evaluators=[...]) → ExperimentResult with per-example scores",
        "effort": "Medium",
        "maps_to": "LangSmith: evaluate() function",
    },
    {
        "priority": 7,
        "name": "Experiment Comparison + Release Decision",
        "what": "Side-by-side score diff for two experiments + PROMOTE/CANARY/HOLD/ROLLBACK output",
        "why": "This is the entire point — turning numbers into a decision that engineers can act on.",
        "how": "compare_experiments(exp_a, exp_b) → delta table + structured release recommendation",
        "effort": "Small",
        "maps_to": "LangSmith: Compare Experiments UI (we built the logic in 05_compare_versions.py)",
    },
]


def print_phase3_plan():
    console.print(Rule("[bold green]Phase 3 — What to Build on Top of Lyzr[/bold green]"))

    for item in PHASE3_PLAN:
        effort_color = {"Small": "green", "Medium": "yellow", "Large": "red"}.get(item["effort"], "white")
        console.print(Panel(
            f"[bold]What:[/bold] {item['what']}\n\n"
            f"[bold]Why first:[/bold] {item['why']}\n\n"
            f"[bold]How:[/bold] {item['how']}\n\n"
            f"[bold]Maps to LangSmith:[/bold] [dim]{item['maps_to']}[/dim]\n"
            f"[bold]Effort:[/bold] [{effort_color}]{item['effort']}[/{effort_color}]",
            title=f"[bold cyan]#{item['priority']} — {item['name']}[/bold cyan]",
            border_style="cyan",
        ))


# ─────────────────────────────────────────────────────────────────────────────
#  ARCHITECTURE DIAGRAM (ASCII)
# ─────────────────────────────────────────────────────────────────────────────

def print_architecture():
    console.print(Rule("[bold magenta]What You're Building on Top of Lyzr[/bold magenta]"))
    console.print(Panel(
        """
  ╔══════════════════════════════════════════════════════════════════╗
  ║          YOUR PRODUCT  (Traces + EvalOps layer)                  ║
  ║                                                                  ║
  ║  ┌────────────────┐  ┌──────────────┐  ┌──────────────────────┐ ║
  ║  │  Trace Access  │  │ Feedback API │  │   Dataset Store      │ ║
  ║  │  list_runs()   │  │ score(run_id)│  │  create / version    │ ║
  ║  │  get_run()     │  │              │  │  import / export     │ ║
  ║  └────────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ ║
  ║           │                 │                     │              ║
  ║  ┌────────▼─────────────────▼─────────────────────▼───────────┐  ║
  ║  │                  Trace → Dataset Pipeline                   │  ║
  ║  │      "Save this production trace as an eval example"        │  ║
  ║  └─────────────────────────────┬───────────────────────────────┘  ║
  ║                                │                                  ║
  ║  ┌─────────────────────────────▼───────────────────────────────┐  ║
  ║  │               Custom Evaluator Registry                     │  ║
  ║  │   @evaluator def correctness(run, example) → score         │  ║
  ║  └─────────────────────────────┬───────────────────────────────┘  ║
  ║                                │                                  ║
  ║  ┌─────────────────────────────▼───────────────────────────────┐  ║
  ║  │                   Experiment Runner                         │  ║
  ║  │   evaluate(agent_fn, dataset, evaluators) → results        │  ║
  ║  └─────────────────────────────┬───────────────────────────────┘  ║
  ║                                │                                  ║
  ║  ┌─────────────────────────────▼───────────────────────────────┐  ║
  ║  │          Experiment Comparison + Release Decision            │  ║
  ║  │     PROMOTE  |  CANARY  |  HOLD  |  ROLLBACK               │  ║
  ║  └─────────────────────────────────────────────────────────────┘  ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║          LYZR PLATFORM  (existing — do not rebuild)              ║
  ║                                                                  ║
  ║  Agent Builder  │  Workflows  │  Governance  │  Memory           ║
  ║  Guardrails     │  OTel Traces│  RAI Evals   │  Studio UI        ║
  ╚══════════════════════════════════════════════════════════════════╝
        """,
        title="[bold magenta]System Architecture[/bold magenta]",
        border_style="magenta",
    ))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Phase 2 — Step 3[/bold cyan]\n"
        "Full Gap Analysis + Phase 3 Build Plan\n\n"
        "[dim]This is the definitive audit that drives what you build next.[/dim]",
        border_style="cyan"
    ))

    print_feature_matrix()
    print_gap_summary()
    print_architecture()
    print_phase3_plan()

    console.print(Panel(
        "[bold green]Phase 2 Complete![/bold green]\n\n"
        "You now have:\n"
        "  ✓ A rebuilt Lyzr agent (same tools, different SDK)\n"
        "  ✓ An SDK probe confirming exactly which methods are missing\n"
        "  ✓ A full 20-feature gap matrix with evidence\n"
        "  ✓ A prioritised 7-item Phase 3 build list\n\n"
        "[bold yellow]Phase 3:[/bold yellow] Build the EvalOps layer on top of Lyzr —\n"
        "starting with the Trace Access API, then Feedback API,\n"
        "then Dataset Store, then everything else.\n\n"
        "[dim]Every piece you build maps directly to a gap confirmed here.[/dim]",
        border_style="green",
        title="[bold]Summary[/bold]"
    ))
