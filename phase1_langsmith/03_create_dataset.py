"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 3 — Turn Production Traces into an Eval Dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  - The full "traces → dataset" pipeline (core EvalOps concept)
  - How to query runs by project, tags, and feedback
  - How to add reference outputs (ground-truth answers) to examples
  - How a LangSmith Dataset is structured
  - Why datasets are the foundation of safe release decisions

CONCEPT — Why do we need datasets from traces?
  ┌──────────────────────────────────────────────────────────────────┐
  │  Hand-written test sets go stale and don't reflect real usage    │
  │                                                                  │
  │  Production traces capture REAL user queries — they show:        │
  │    • Edge cases you never thought to write                       │
  │    • The exact phrasing users use (not what devs assume)         │
  │    • Failures that happened in prod (low feedback scores)        │
  │                                                                  │
  │  Dataset = the "ground truth" you evaluate new versions against  │
  └──────────────────────────────────────────────────────────────────┘

DATASET STRUCTURE in LangSmith:
  Dataset
  └── Example[]
        ├── inputs  : { "input": "How do I reset my password?" }
        ├── outputs : { "output": "Click Forgot Password on login page..." }  ← reference answer
        └── metadata: { "source_run_id": "...", "feedback_score": 1.0 }

HOW TO RUN:
  python phase1_langsmith/03_create_dataset.py

PREREQ: Run 02_langsmith_tracing.py first so there are traces to pull.
"""

import os
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from langsmith import Client

load_dotenv()
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  REFERENCE ANSWERS (ground truth)
#
#  In a real workflow, these come from:
#    a) A human reviewer who reads the trace and writes the ideal answer
#    b) An existing documentation page
#    c) A "golden" agent run that was manually verified
#
#  For this exercise, we write them manually so you can see the structure.
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_ANSWERS = {
    "How do I update my billing information?": (
        "To update your billing information, go to Settings → Billing in your account. "
        "To change your credit card, navigate to Settings → Payment Methods. "
        "Invoices are generated on the 1st of each month."
    ),
    "What is the API rate limit for the Pro plan?": (
        "The Pro plan allows 10,000 API calls per day. "
        "If you exceed this, you will receive an HTTP 429 error. "
        "You can request a temporary increase via Settings → API → Request Increase."
    ),
    "I need to set up SSO with Okta for my enterprise.": (
        "SSO is available on Enterprise plans. Okta is a supported provider along with "
        "Azure AD and Google Workspace. Follow the configuration guide at docs.example.com/sso."
    ),
    "What's the status of ticket TKT-003?": (
        "Ticket TKT-003 is currently in Pending status. "
        "The issue is SSO configuration help and it was opened 1 day ago."
    ),
    "My app keeps getting HTTP 429 errors, can you help?": (
        "HTTP 429 means you've exceeded your API rate limit. "
        "The Pro plan limit is 10,000 calls/day. "
        "You can request a temporary increase via Settings → API → Request Increase."
    ),
    "I haven't received my password reset email.": (
        "Password reset emails are sent within 2 minutes. "
        "Please check your spam folder. "
        "If the issue persists, I recommend creating a support ticket."
    ),
    "How do I export my data to JSON format?": (
        "Go to Settings → Data → Export to download your data. "
        "JSON, CSV, and Parquet formats are supported. "
        "Exports are processed asynchronously and emailed to you when ready."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 1: Create dataset from hard-coded examples
#
#  USE THIS WHEN: you want a clean starting dataset before you have any traces
# ─────────────────────────────────────────────────────────────────────────────

def create_seed_dataset(client: Client, dataset_name: str) -> str:
    """
    Creates a 'seed' dataset from our predefined Q&A pairs.
    This is your starting point — a clean, manually verified set.
    """
    console.print(Rule("[bold cyan]Creating Seed Dataset[/bold cyan]"))

    # Delete existing dataset if it exists (for re-runs during learning)
    existing = [d for d in client.list_datasets() if d.name == dataset_name]
    if existing:
        client.delete_dataset(dataset_id=existing[0].id)
        console.print(f"[dim]Deleted existing dataset '{dataset_name}'[/dim]")

    # Create fresh dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=(
            "Tech Support Agent evaluation dataset. "
            "Created from curated Q&A pairs for Phase 1 learning."
        ),
    )
    console.print(f"[green]✓ Dataset created:[/green] {dataset.name} (id: {str(dataset.id)[:12]}...)")

    # Add examples (input + reference output pairs)
    examples_created = 0
    for question, reference_answer in REFERENCE_ANSWERS.items():
        client.create_example(
            inputs={"input": question},
            outputs={"output": reference_answer},
            dataset_id=dataset.id,
            metadata={
                "source": "manual_curation",
                "category": _categorize(question),
            },
        )
        examples_created += 1

    console.print(f"[green]✓ Added {examples_created} examples to dataset[/green]")
    return str(dataset.id)


def _categorize(question: str) -> str:
    """Simple category tag for dataset examples."""
    q = question.lower()
    if "billing" in q or "payment" in q:      return "billing"
    if "api" in q or "429" in q or "rate" in q: return "api"
    if "sso" in q or "okta" in q:             return "sso"
    if "ticket" in q or "tkt" in q:           return "ticketing"
    if "password" in q or "reset" in q:       return "auth"
    if "export" in q or "data" in q:          return "data"
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 2: Enrich dataset with production traces
#
#  USE THIS WHEN: you have real traces and want to add them to your dataset
#  This is the "production traces → dataset" pipeline that's core to EvalOps
# ─────────────────────────────────────────────────────────────────────────────

def enrich_from_traces(client: Client, dataset_name: str, project_name: str):
    """
    Pulls runs from LangSmith that had NEGATIVE feedback (score=0.0)
    and adds them to the dataset so we can track improvement.

    WHY negative feedback runs?
      - These are the FAILURES — queries where the agent didn't satisfy the user
      - Adding them to the dataset means future versions must do better on them
      - This is how you close the feedback loop: prod failure → eval case
    """
    console.print(Rule("[bold yellow]Enriching Dataset from Production Traces[/bold yellow]"))

    dataset = next((d for d in client.list_datasets() if d.name == dataset_name), None)
    if not dataset:
        console.print("[red]Dataset not found. Run create_seed_dataset first.[/red]")
        return

    # Fetch runs with low feedback scores
    # In LangSmith, you filter runs and then check their feedback
    runs = list(client.list_runs(
        project_name=project_name,
        run_type="chain",
        limit=50,
    ))

    added = 0
    skipped = 0

    for run in runs:
        if not run.inputs or "input" not in run.inputs:
            skipped += 1
            continue

        input_text = run.inputs["input"]
        output_text = run.outputs.get("output", "") if run.outputs else ""

        # Check if this run already exists in our dataset (avoid duplicates)
        existing_inputs = {
            str(ex.inputs.get("input", ""))
            for ex in client.list_examples(dataset_id=dataset.id)
        }
        if input_text in existing_inputs:
            skipped += 1
            continue

        # Add this production run as a new dataset example
        # Note: output is the agent's actual answer (not ground truth yet)
        # A human reviewer would later update the 'outputs' with the ideal answer
        client.create_example(
            inputs={"input": input_text},
            outputs={"output": output_text},  # agent's answer — to be reviewed
            dataset_id=dataset.id,
            metadata={
                "source": "production_trace",
                "source_run_id": str(run.id),
                "needs_review": True,          # flag for human reviewers
                "run_status": run.status,
            },
        )
        added += 1

    console.print(f"[green]✓ Added {added} production traces to dataset[/green]")
    console.print(f"[dim]  Skipped {skipped} (duplicates or missing inputs)[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
#  INSPECT THE DATASET
# ─────────────────────────────────────────────────────────────────────────────

def inspect_dataset(client: Client, dataset_name: str):
    """
    Prints a summary of all examples in the dataset.
    This is what you'd review before running evaluations.
    """
    console.print(Rule("[bold cyan]Dataset Inspection[/bold cyan]"))

    dataset = next((d for d in client.list_datasets() if d.name == dataset_name), None)
    if not dataset:
        console.print("[red]Dataset not found.[/red]")
        return

    examples = list(client.list_examples(dataset_id=dataset.id))

    table = Table(
        title=f"Dataset: {dataset_name}  ({len(examples)} examples)",
        border_style="blue",
        show_lines=True,
    )
    table.add_column("#", width=3)
    table.add_column("Input (truncated)", width=45)
    table.add_column("Category", width=12)
    table.add_column("Source", width=16)
    table.add_column("Reference Answer?", width=17)

    for i, ex in enumerate(examples, 1):
        inp = str(ex.inputs.get("input", ""))[:43] + "…"
        category = ex.metadata.get("category", "—") if ex.metadata else "—"
        source   = ex.metadata.get("source", "—") if ex.metadata else "—"
        has_ref  = "[green]✓ Yes[/green]" if ex.outputs else "[red]✗ No[/red]"
        table.add_row(str(i), inp, category, source, has_ref)

    console.print(table)

    # Export as JSON for inspection
    export = [
        {
            "id": str(ex.id),
            "input": ex.inputs,
            "reference_output": ex.outputs,
            "metadata": ex.metadata,
        }
        for ex in examples
    ]
    export_path = "phase1_langsmith/dataset_export.json"
    with open(export_path, "w") as f:
        json.dump(export, f, indent=2)
    console.print(f"\n[dim]Dataset exported to {export_path} for local review.[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATASET_NAME  = "tech-support-agent-v1"
    PROJECT_NAME  = os.getenv("LANGCHAIN_PROJECT", "lyzr-evalops-phase1")

    console.print(Panel.fit(
        "[bold cyan]Phase 1 — Step 3[/bold cyan]\n"
        "Traces → Evaluation Dataset\n\n"
        "[dim]We will:\n"
        "  1. Create a seed dataset from curated Q&A pairs\n"
        "  2. Enrich it with real production traces\n"
        "  3. Inspect the full dataset before evaluation[/dim]",
        border_style="cyan"
    ))

    client = Client()

    # Step 3a — seed dataset
    create_seed_dataset(client, DATASET_NAME)

    # Step 3b — enrich with production traces from step 2
    enrich_from_traces(client, DATASET_NAME, PROJECT_NAME)

    # Step 3c — inspect
    inspect_dataset(client, DATASET_NAME)

    console.print(Panel(
        "[bold green]Dataset ready![/bold green]\n\n"
        "In LangSmith UI → Datasets & Testing → [bold]tech-support-agent-v1[/bold]\n"
        "  • Browse all examples\n"
        "  • Edit reference answers\n"
        "  • Add/remove examples\n\n"
        "[bold yellow]Next:[/bold yellow] Run [bold]04_run_evaluations.py[/bold] "
        "to score the agent against this dataset using LLM-as-a-judge.",
        border_style="green"
    ))
