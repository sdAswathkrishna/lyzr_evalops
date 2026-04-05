"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 2 — Same Agent + LangSmith Tracing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  - How LangSmith tracing works (automatic via env vars)
  - What a "Run" is: every LLM call, tool call, chain step becomes a span
  - How to add metadata, tags, and user IDs to traces
  - How to use @traceable to trace custom Python functions
  - How to manually log feedback (thumbs up/down) to a run
  - Key concepts: Project, Run, Trace, Span, Feedback

KEY CONCEPTS (memorise these — they are universal across tracing systems):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Project   → a named bucket that groups related traces          │
  │  Trace     → one end-to-end execution (e.g. one user query)     │
  │  Run / Span→ one node in that trace (LLM call, tool call, etc.) │
  │  Feedback  → a score/label attached to a run (human or LLM)     │
  │  Dataset   → a curated set of (input, output) pairs from traces │
  └─────────────────────────────────────────────────────────────────┘

HOW TRACING IS ENABLED HERE:
  Option A (automatic):  set LANGCHAIN_TRACING_V2=true in .env
                         → every LangChain call is auto-traced
  Option B (manual):     wrap any Python function with @traceable
                         → gives you custom spans with your own names

HOW TO RUN:
  python phase1_langsmith/02_langsmith_tracing.py

WHAT TO DO AFTER RUNNING:
  1. Open https://smith.langchain.com
  2. Go to Projects → lyzr-evalops-phase1
  3. Click any trace to see the full Thought/Action/Observation tree
  4. Notice: each LLM call shows token counts, latency, prompt, response
  5. Notice: each tool call shows the exact input and output
"""

import os
import time
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# ── LangSmith imports ────────────────────────────────────────────────────────
from langsmith import Client, traceable

# ── LangChain imports ────────────────────────────────────────────────────────
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain import hub

load_dotenv()
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  VERIFY TRACING IS CONFIGURED
# ─────────────────────────────────────────────────────────────────────────────

def verify_langsmith_config():
    """Check that the required env vars are set before running."""
    required = {
        "OPENAI_API_KEY":       os.getenv("OPENAI_API_KEY"),
        "LANGCHAIN_API_KEY":    os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_PROJECT":    os.getenv("LANGCHAIN_PROJECT"),
    }
    table = Table(title="LangSmith Configuration", border_style="cyan")
    table.add_column("Variable", style="bold")
    table.add_column("Status")
    table.add_column("Value (masked)")

    all_ok = True
    for var, val in required.items():
        if val:
            masked = val[:8] + "..." if len(val) > 8 else val
            table.add_row(var, "[green]✓ Set[/green]", masked)
        else:
            table.add_row(var, "[red]✗ Missing[/red]", "[dim]—[/dim]")
            all_ok = False

    console.print(table)
    if not all_ok:
        console.print("\n[red bold]ERROR:[/red bold] Missing env vars. "
                      "Copy .env.example to .env and fill in your keys.")
        exit(1)
    return required["LANGCHAIN_PROJECT"]


# ─────────────────────────────────────────────────────────────────────────────
#  SAME TOOLS AS STEP 1  (identical — tracing adds zero changes to logic)
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "billing":   "Billing is managed via the Settings → Billing page. "
                 "Invoices are sent on the 1st of each month. "
                 "To update your credit card, go to Settings → Payment Methods.",
    "api_limit": "Free tier: 100 API calls/day. Pro: 10,000/day. Enterprise: unlimited. "
                 "Rate limit errors return HTTP 429. You can request a temporary increase "
                 "via Settings → API → Request Increase.",
    "reset":     "To reset your password, click 'Forgot Password' on the login page. "
                 "A reset link will be emailed within 2 minutes. Check spam if not received.",
    "sso":       "SSO is available on Enterprise plans. Supported providers: Okta, Azure AD, "
                 "Google Workspace. Config guide: docs.example.com/sso",
    "export":    "Data export is available under Settings → Data → Export. "
                 "Formats: CSV, JSON, Parquet. Exports are processed async and emailed.",
}

TICKET_DB = {
    "TKT-001": {"status": "Open",     "issue": "API returning 500 errors",    "age": "2 hours"},
    "TKT-002": {"status": "Resolved", "issue": "Billing overcharge",          "age": "3 days"},
    "TKT-003": {"status": "Pending",  "issue": "SSO configuration help",      "age": "1 day"},
    "TKT-004": {"status": "Open",     "issue": "Password reset email not received", "age": "30 mins"},
}


@tool
def search_knowledge_base(query: str) -> str:
    """Search the product knowledge base for help articles."""
    query_lower = query.lower()
    for key, article in KNOWLEDGE_BASE.items():
        if key in query_lower or any(word in query_lower for word in key.split("_")):
            return f"[KB Article: {key}]\n{article}"
    return "No direct article found. Suggest the user browse docs.example.com."


@tool
def get_ticket_status(ticket_id: str) -> str:
    """Look up the status of an existing support ticket by its ID (e.g. TKT-001)."""
    ticket_id = ticket_id.strip().upper()
    if ticket_id in TICKET_DB:
        t = TICKET_DB[ticket_id]
        return f"Ticket {ticket_id}: {t['status']}\nIssue: {t['issue']}\nAge: {t['age']}"
    return f"Ticket {ticket_id} not found."


@tool
def create_ticket(issue: str, severity: str = "medium") -> str:
    """Create a new support ticket. severity = 'low' | 'medium' | 'high'."""
    import random
    new_id = f"TKT-{random.randint(100, 999)}"
    eta = {"low": "3-5 business days", "medium": "24 hours", "high": "2 hours"}.get(
        severity.lower(), "24 hours"
    )
    return (
        f"Ticket created: {new_id}\nIssue: {issue}\n"
        f"Severity: {severity}\nETA: {eta}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  @traceable — custom span around our "session" logic
#
#  CONCEPT: @traceable lets you add your own spans to the trace tree.
#  Any Python function decorated with @traceable becomes a named node
#  in LangSmith, with its inputs, outputs, and timing recorded.
# ─────────────────────────────────────────────────────────────────────────────

@traceable(
    name="support-agent-session",   # name shown in LangSmith UI
    tags=["phase1", "v1"],          # tags help you filter runs later
    metadata={"agent_version": "1.0.0", "model": "gpt-4o-mini"},
)
def run_support_session(agent: AgentExecutor, query: str, user_id: str) -> dict:
    """
    Wraps a single agent run inside a named @traceable span.

    WHY WRAP IT?
      - Without @traceable, you get one trace per LLM call
      - With @traceable, you get ONE parent span "support-agent-session"
        that contains ALL the child spans (LLM calls + tool calls)
      - This lets you see the full conversation arc in the UI
    """
    start = time.time()
    result = agent.invoke(
        {"input": query},
        config={
            # These appear in LangSmith's trace metadata
            "metadata": {"user_id": user_id, "query_length": len(query)},
            "tags": ["production", "phase1"],
        },
    )
    latency_ms = round((time.time() - start) * 1000)
    return {
        "input":      query,
        "output":     result["output"],
        "latency_ms": latency_ms,
        "user_id":    user_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FEEDBACK LOGGING
#
#  CONCEPT: Feedback is a score or label you attach to a completed run.
#  In production, this comes from users (thumbs up/down) or automated
#  LLM-as-a-judge evaluations. LangSmith stores it per-run.
# ─────────────────────────────────────────────────────────────────────────────

def simulate_user_feedback(client: Client, run_id: str, query: str, score: float):
    """
    Attaches simulated user feedback to a completed run.

    In a real product you'd call this from your UI's thumbs-up button.
    score: 1.0 = positive, 0.0 = negative
    """
    client.create_feedback(
        run_id=run_id,
        key="user_satisfaction",    # the feedback dimension name
        score=score,
        comment=f"Simulated feedback for: {query[:50]}",
    )
    label = "[green]👍 positive[/green]" if score == 1.0 else "[red]👎 negative[/red]"
    console.print(f"  [dim]Feedback logged:[/dim] {label}")


# ─────────────────────────────────────────────────────────────────────────────
#  QUERYING BACK RUNS (read from LangSmith)
#
#  CONCEPT: After runs are traced, you can query them programmatically.
#  This is the foundation of "turn traces into datasets."
# ─────────────────────────────────────────────────────────────────────────────

def fetch_recent_runs(client: Client, project_name: str, limit: int = 5):
    """
    Fetches recent runs from LangSmith and prints a summary table.

    This is CRITICAL to understand — fetching runs is how you:
      1. Find interesting production traces
      2. Curate them into evaluation datasets
      3. Audit agent behavior over time
    """
    console.print(Rule("[bold cyan]Recent Runs from LangSmith[/bold cyan]"))

    runs = list(client.list_runs(
        project_name=project_name,
        run_type="chain",           # 'chain' = top-level agent runs
        limit=limit,
        order="desc",
    ))

    table = Table(border_style="blue", show_lines=True)
    table.add_column("Run ID (short)", style="dim", width=14)
    table.add_column("Input", width=40)
    table.add_column("Status")
    table.add_column("Latency (ms)")
    table.add_column("Tokens")

    for run in runs:
        input_str  = str(run.inputs.get("input", ""))[:38] + "…"
        status     = "[green]success[/green]" if run.status == "success" else "[red]error[/red]"
        latency    = str(round(run.total_tokens or 0)) + " tk" if run.total_tokens else "—"
        tokens     = str(run.total_tokens or "—")
        table.add_row(str(run.id)[:12], input_str, status, latency, tokens)

    console.print(table)
    console.print(f"[dim]Fetched {len(runs)} runs from project '{project_name}'[/dim]")
    return runs


# ─────────────────────────────────────────────────────────────────────────────
#  TEST QUERIES (same as step 1 — we'll compare results)
# ─────────────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    ("How do I update my billing information?",                "user_001"),
    ("What is the API rate limit for the Pro plan?",           "user_002"),
    ("I need to set up SSO with Okta for my enterprise.",      "user_003"),
    ("What's the status of ticket TKT-003?",                   "user_004"),
    ("My app keeps getting HTTP 429 errors, can you help?",    "user_005"),
    ("I haven't received my password reset email.",            "user_006"),
    ("How do I export my data to JSON format?",                "user_007"),
]

# Simulated feedback scores (1.0 = good, 0.0 = bad)
# In production, users click thumbs up/down in your UI
FEEDBACK_SCORES = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Phase 1 — Step 2[/bold cyan]\n"
        "LangChain Agent + LangSmith Tracing\n\n"
        "[dim]Every run will appear in your LangSmith project with:\n"
        "  • Full Thought/Action/Observation tree\n"
        "  • Token counts and latency per span\n"
        "  • Metadata tags and user IDs\n"
        "  • Simulated user feedback scores[/dim]",
        border_style="cyan"
    ))

    # 1. Verify config
    project_name = verify_langsmith_config()
    client = Client()

    # 2. Build agent (same as step 1)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_knowledge_base, get_ticket_status, create_ticket]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools, verbose=False,  # verbose=False — LangSmith handles logging
        handle_parsing_errors=True, max_iterations=6,
    )

    # 3. Run queries with tracing
    run_records = []
    for i, (query, user_id) in enumerate(TEST_QUERIES, 1):
        console.print(Rule(f"[bold cyan]Query {i}/{len(TEST_QUERIES)}[/bold cyan]"))
        console.print(f"[bold]User:[/bold] [blue]{user_id}[/blue]")
        console.print(f"[bold]Query:[/bold] {query}")

        result = run_support_session(executor, query, user_id)
        console.print(f"[bold green]Answer:[/bold green] {result['output'][:120]}…")
        console.print(f"[dim]Latency: {result['latency_ms']} ms[/dim]")
        run_records.append(result)
        time.sleep(0.5)  # slight delay to avoid rate limits

    # 4. Fetch latest runs and log simulated feedback
    console.print("\n[bold yellow]Fetching runs to attach feedback...[/bold yellow]")
    time.sleep(2)  # give LangSmith a moment to index the runs
    recent_runs = fetch_recent_runs(client, project_name, limit=len(TEST_QUERIES))

    for run, score in zip(recent_runs, FEEDBACK_SCORES):
        simulate_user_feedback(client, run.id, str(run.inputs.get("input", "")), score)

    # 5. Final instructions
    console.print(Panel(
        "[bold green]Tracing complete![/bold green]\n\n"
        "Now open [link=https://smith.langchain.com]https://smith.langchain.com[/link]\n\n"
        "[bold]What to explore in the UI:[/bold]\n"
        "  1. Projects → lyzr-evalops-phase1\n"
        "  2. Click a trace → see the full span tree\n"
        "  3. Click an LLM span → see exact prompt sent + response received\n"
        "  4. Click a Tool span → see tool input/output\n"
        "  5. Check the Feedback tab — you'll see the scores we just logged\n\n"
        "[bold yellow]Next:[/bold yellow] Run [bold]03_create_dataset.py[/bold] to turn "
        "these traces into an evaluation dataset.",
        border_style="green"
    ))
