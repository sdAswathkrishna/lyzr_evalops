"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE 2 — STEP 2: What Lyzr Actually Has
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THIS DOES:
  Systematically probes Lyzr's Python SDK to find every method
  that relates to tracing, feedback, datasets, and evaluation.

  This is the "reality check" — we look at what the SDK actually
  exposes vs what the marketing docs claim.

PROBING STRATEGY:
  1. Inspect the Studio class — list all public methods
  2. Try to find trace-related methods
  3. Try to find feedback-related methods
  4. Try to find dataset-related methods
  5. Try to find evaluation-related methods
  6. Attempt calls that SHOULD exist if Lyzr had LangSmith parity
     → document which ones succeed vs raise AttributeError / NotImplementedError

HOW TO RUN:
  python phase2_lyzr_audit/02_what_lyzr_has.py
  (Works even without LYZR_API_KEY — SDK inspection is local)
"""

import os
import inspect
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

load_dotenv()
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: INSPECT A CLASS AND CATEGORISE ITS PUBLIC METHODS
# ─────────────────────────────────────────────────────────────────────────────

def inspect_class(cls, class_name: str):
    """List all public methods of a class with their signatures."""
    console.print(Rule(f"[bold cyan]{class_name} — Public API[/bold cyan]"))

    methods = [
        (name, func)
        for name, func in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]

    if not methods:
        methods = [
            (name, getattr(cls, name))
            for name in dir(cls)
            if not name.startswith("_") and callable(getattr(cls, name, None))
        ]

    table = Table(border_style="blue", show_lines=True)
    table.add_column("Method", style="bold", width=30)
    table.add_column("Signature / Parameters", width=55)

    for name, func in sorted(methods):
        try:
            sig = str(inspect.signature(func))
        except (ValueError, TypeError):
            sig = "(signature not available)"
        table.add_row(name, sig[:53])

    console.print(table)
    console.print(f"[dim]Total public methods: {len(methods)}[/dim]\n")
    return [name for name, _ in methods]


# ─────────────────────────────────────────────────────────────────────────────
#  PROBE 1 — Studio class (the main ADK entry point)
# ─────────────────────────────────────────────────────────────────────────────

def probe_studio_class():
    console.print(Rule("[bold yellow]PROBE 1: lyzr_adk.Studio[/bold yellow]"))
    try:
        from lyzr import Studio              # package: lyzr-adk, module: lyzr
        methods = inspect_class(Studio, "lyzr_adk.Studio")

        # Classify methods by category
        categories = {
            "Agent lifecycle": [m for m in methods if any(
                k in m for k in ["agent", "create", "update", "delete", "get", "list"])],
            "Knowledge Base":  [m for m in methods if any(
                k in m for k in ["knowledge", "kb", "document", "embed"])],
            "Tracing":         [m for m in methods if any(
                k in m for k in ["trace", "span", "run", "log", "observ"])],
            "Feedback":        [m for m in methods if any(
                k in m for k in ["feedback", "score", "rate", "rating"])],
            "Dataset":         [m for m in methods if any(
                k in m for k in ["dataset", "example", "sample", "eval_data"])],
            "Evaluation":      [m for m in methods if any(
                k in m for k in ["eval", "judge", "assess", "metric"])],
        }

        cat_table = Table(title="Method Categories", border_style="yellow")
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("Methods Found")
        cat_table.add_column("Status")

        for cat, found in categories.items():
            if found:
                cat_table.add_row(cat, ", ".join(found), "[green]✓ Has[/green]")
            else:
                cat_table.add_row(cat, "[dim]none[/dim]", "[red]✗ Missing[/red]")

        console.print(cat_table)

    except ImportError as e:
        console.print(f"[red]Import failed: {e}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
#  PROBE 2 — lyzr_agent_api client
# ─────────────────────────────────────────────────────────────────────────────

def probe_agent_api():
    console.print(Rule("[bold yellow]PROBE 2: lyzr_agent_api.AgentAPI[/bold yellow]"))
    try:
        from lyzr_agent_api.client import AgentAPI
        methods = inspect_class(AgentAPI, "lyzr_agent_api.AgentAPI")

        # Look for EvalOps-relevant methods
        eval_ops_targets = [
            "get_trace", "list_traces", "get_run", "list_runs",
            "add_feedback", "create_feedback", "score_run",
            "create_dataset", "list_datasets", "add_example",
            "run_evaluation", "create_experiment",
        ]

        console.print("\n[bold]Searching for EvalOps methods:[/bold]")
        for target in eval_ops_targets:
            found = any(target.lower() in m.lower() for m in methods)
            status = "[green]✓ Found[/green]" if found else "[red]✗ Missing[/red]"
            console.print(f"  {target:<30} {status}")

    except ImportError as e:
        console.print(f"[red]Import failed: {e}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
#  PROBE 3 — Try to call what SHOULD exist (if LangSmith parity existed)
#
#  These calls will fail — that's the point.
#  We document exactly what error each one raises.
# ─────────────────────────────────────────────────────────────────────────────

def probe_missing_calls():
    """
    Attempt to call EvalOps methods that LangSmith has but Lyzr doesn't.
    This makes the gaps concrete and runnable — not just theoretical.
    """
    console.print(Rule("[bold red]PROBE 3: Attempting LangSmith-Equivalent Calls on Lyzr[/bold red]"))
    console.print("[dim]These calls intentionally test whether Lyzr has the same capabilities.[/dim]\n")

    results = []

    # ── Test A: Get trace back from a run ────────────────────────────────────
    console.print("[bold]Test A — Can we get a trace ID from an agent run?[/bold]")
    console.print("[dim]In LangSmith: run_id is returned after agent.invoke()[/dim]")
    try:
        from lyzr import Studio              # package: lyzr-adk, module: lyzr
        api_key = os.getenv("LYZR_API_KEY", "fake-key-for-testing")
        studio = Studio(api_key=api_key)
        # Attempt to find a get_trace or list_runs method
        has_trace = hasattr(studio, "get_trace") or hasattr(studio, "list_runs")
        if has_trace:
            result = "✓ Method exists"
        else:
            result = "✗ No get_trace / list_runs method on Studio"
    except Exception as e:
        result = f"✗ Error: {type(e).__name__}: {e}"
    console.print(f"  Result: [yellow]{result}[/yellow]\n")
    results.append(("Get trace ID from run", result))

    # ── Test B: Attach programmatic feedback ─────────────────────────────────
    console.print("[bold]Test B — Can we attach a score to a run programmatically?[/bold]")
    console.print("[dim]In LangSmith: client.create_feedback(run_id, key='correctness', score=0.9)[/dim]")
    try:
        from lyzr import Studio              # package: lyzr-adk, module: lyzr
        api_key = os.getenv("LYZR_API_KEY", "fake-key-for-testing")
        studio = Studio(api_key=api_key)
        has_feedback = (
            hasattr(studio, "create_feedback") or
            hasattr(studio, "add_feedback")    or
            hasattr(studio, "log_feedback")
        )
        result = "✓ Method exists" if has_feedback else "✗ No feedback method on Studio"
    except Exception as e:
        result = f"✗ Error: {type(e).__name__}: {e}"
    console.print(f"  Result: [yellow]{result}[/yellow]\n")
    results.append(("Attach score to run (Feedback API)", result))

    # ── Test C: Create a dataset ──────────────────────────────────────────────
    console.print("[bold]Test C — Can we create an evaluation dataset?[/bold]")
    console.print("[dim]In LangSmith: client.create_dataset(name='my-dataset')[/dim]")
    try:
        from lyzr import Studio              # package: lyzr-adk, module: lyzr
        api_key = os.getenv("LYZR_API_KEY", "fake-key-for-testing")
        studio = Studio(api_key=api_key)
        has_dataset = (
            hasattr(studio, "create_dataset") or
            hasattr(studio, "add_dataset")    or
            hasattr(studio, "list_datasets")
        )
        result = "✓ Method exists" if has_dataset else "✗ No dataset method on Studio"
    except Exception as e:
        result = f"✗ Error: {type(e).__name__}: {e}"
    console.print(f"  Result: [yellow]{result}[/yellow]\n")
    results.append(("Create evaluation dataset", result))

    # ── Test D: Run LLM-as-judge evaluation ──────────────────────────────────
    console.print("[bold]Test D — Can we run a custom LLM-as-judge evaluator?[/bold]")
    console.print("[dim]In LangSmith: evaluate(runner, data=dataset, evaluators=[my_fn])[/dim]")
    try:
        from lyzr import Studio              # package: lyzr-adk, module: lyzr
        api_key = os.getenv("LYZR_API_KEY", "fake-key-for-testing")
        studio = Studio(api_key=api_key)
        has_eval = (
            hasattr(studio, "evaluate")       or
            hasattr(studio, "run_evaluation") or
            hasattr(studio, "create_evaluator")
        )
        result = "✓ Method exists" if has_eval else "✗ No evaluate / run_evaluation method on Studio"
    except Exception as e:
        result = f"✗ Error: {type(e).__name__}: {e}"
    console.print(f"  Result: [yellow]{result}[/yellow]\n")
    results.append(("Run custom LLM-as-judge evaluation", result))

    # ── Test E: Compare two experiments ──────────────────────────────────────
    console.print("[bold]Test E — Can we compare two agent experiments on the same dataset?[/bold]")
    console.print("[dim]In LangSmith: evaluate() twice with different runners, then compare in UI[/dim]")
    try:
        from lyzr import Studio              # package: lyzr-adk, module: lyzr
        api_key = os.getenv("LYZR_API_KEY", "fake-key-for-testing")
        studio = Studio(api_key=api_key)
        has_compare = (
            hasattr(studio, "compare_experiments") or
            hasattr(studio, "compare_runs")        or
            hasattr(studio, "create_experiment")
        )
        result = "✓ Method exists" if has_compare else "✗ No experiment comparison method on Studio"
    except Exception as e:
        result = f"✗ Error: {type(e).__name__}: {e}"
    console.print(f"  Result: [yellow]{result}[/yellow]\n")
    results.append(("Compare two experiments (A/B)", result))

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(Rule("[bold]EvalOps Call Probe Summary[/bold]"))
    summary = Table(border_style="red", show_lines=True)
    summary.add_column("Capability Tested", style="bold")
    summary.add_column("Result")
    for cap, res in results:
        color = "green" if "✓" in res else "red"
        summary.add_row(cap, f"[{color}]{res}[/{color}]")
    console.print(summary)

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PROBE 4 — What Lyzr's RAI actually does (the strength)
# ─────────────────────────────────────────────────────────────────────────────

def describe_lyzr_strengths():
    """
    Document what Lyzr actually does well vs LangChain.
    Fair audit = document both gaps AND strengths.
    """
    console.print(Rule("[bold green]What Lyzr Has That LangChain Doesn't[/bold green]"))

    strengths = [
        ("Built-in RAI Guardrails",
         "Toxicity, PII (9 types), hallucination, prompt injection,\n"
         "NSFW, bias, secrets masking — ON by default, no code needed.\n"
         "LangChain: you build this manually as tools or nodes."),

        ("Trained ML safety models",
         "Lyzr uses purpose-trained ML classifiers for harmful content,\n"
         "not LLM-as-judge. More deterministic and cheaper for safety.\n"
         "LangChain: no equivalent out of the box."),

        ("Memory built-in",
         "agent = studio.create_agent(memory=10) — done.\n"
         "LangChain: requires ConversationBufferMemory + chain wiring."),

        ("Multi-LLM support",
         "Switch between OpenAI, Anthropic, Gemini, Groq, Bedrock\n"
         "with one parameter change: provider='anthropic'.\n"
         "LangChain: need different import per provider."),

        ("No-code Agent Builder UI",
         "Non-technical users can build, test, and deploy agents\n"
         "in Studio without writing any code.\n"
         "LangSmith: assumes developers, no no-code builder."),

        ("Operation Waterfall (OTel)",
         "Built-in span visualization backed by OpenTelemetry.\n"
         "Shows tool latency contributions, model calls, memory ops.\n"
         "LangChain: needs LangSmith or custom OTEL setup."),

        ("Agent Simulation Engine",
         "Persona × Scenario matrix generates test cases automatically.\n"
         "Iterative hardening: eval → identify gaps → improve → re-eval.\n"
         "LangSmith: you must write test cases manually."),
    ]

    for title, description in strengths:
        console.print(Panel(
            description,
            title=f"[bold green]✓ {title}[/bold green]",
            border_style="green",
        ))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Phase 2 — Step 2[/bold cyan]\n"
        "Probing Lyzr's Actual SDK Capabilities\n\n"
        "[dim]We inspect every public method on Lyzr's SDK classes\n"
        "and attempt calls that SHOULD exist if Lyzr had LangSmith parity.\n"
        "This makes the gaps concrete, not theoretical.[/dim]",
        border_style="cyan"
    ))

    probe_studio_class()
    probe_agent_api()
    probe_missing_calls()
    describe_lyzr_strengths()

    console.print(
        "\n[bold yellow]Next:[/bold yellow] Run [bold]03_gap_analysis.py[/bold] "
        "for the full structured comparison and Phase 3 build plan."
    )
