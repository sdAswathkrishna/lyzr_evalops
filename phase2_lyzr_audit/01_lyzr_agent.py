"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE 2 — STEP 1: Same Agent, Built in Lyzr ADK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THIS DOES:
  Rebuilds the exact same Tech Support Agent from Phase 1
  using Lyzr's ADK instead of LangChain.

  This lets you feel the difference in:
    - How you define tools (Python functions vs @tool decorator)
    - How you configure the agent (Studio object vs AgentExecutor)
    - What you get out of the box (RAI guardrails, memory)
    - What you DON'T get (tracing control, feedback API)

LYZR ADK MENTAL MODEL:
  ┌──────────────────────────────────────────────────────────┐
  │  Studio          → the top-level client (like LangSmith  │
  │                    Client + AgentExecutor combined)       │
  │  studio.create_agent() → creates + registers the agent   │
  │  agent.add_tool()      → attaches a Python function      │
  │  agent.run(message)    → runs one turn (no ReAct loop    │
  │                          shown — it's internal)           │
  │  RAI Policy            → guardrails baked in by default  │
  └──────────────────────────────────────────────────────────┘

PREREQ:
  Add LYZR_API_KEY to your .env file.
  Get it from: https://studio.lyzr.ai → Settings → API Keys

HOW TO RUN:
  python phase2_lyzr_audit/01_lyzr_agent.py
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.columns import Columns

load_dotenv()
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  LYZR SDK IMPORT
# ─────────────────────────────────────────────────────────────────────────────
try:
    from lyzr import Studio          # package: lyzr-adk, module: lyzr
    LYZR_AVAILABLE = True
except ImportError:
    LYZR_AVAILABLE = False
    console.print("[red]lyzr-adk not installed. Run: pip install lyzr-adk[/red]")

# ─────────────────────────────────────────────────────────────────────────────
#  SAME MOCK DATA AS PHASE 1
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "billing":   "Billing is managed via the Settings → Billing page. "
                 "Invoices are sent on the 1st of each month. "
                 "To update your credit card, go to Settings → Payment Methods.",
    "api_limit": "Free tier: 100 API calls/day. Pro: 10,000/day. Enterprise: unlimited. "
                 "Rate limit errors return HTTP 429. Request a temporary increase "
                 "via Settings → API → Request Increase.",
    "reset":     "To reset your password, click 'Forgot Password' on the login page. "
                 "A reset link will be emailed within 2 minutes. Check spam if not received.",
    "sso":       "SSO is available on Enterprise plans. Supported: Okta, Azure AD, "
                 "Google Workspace. Config guide: docs.example.com/sso",
    "export":    "Data export: Settings → Data → Export. "
                 "Formats: CSV, JSON, Parquet. Exports are async and emailed.",
}
TICKET_DB = {
    "TKT-001": {"status": "Open",     "issue": "API returning 500 errors",    "age": "2 hours"},
    "TKT-002": {"status": "Resolved", "issue": "Billing overcharge",          "age": "3 days"},
    "TKT-003": {"status": "Pending",  "issue": "SSO configuration help",      "age": "1 day"},
    "TKT-004": {"status": "Open",     "issue": "Password reset not received", "age": "30 mins"},
}

# ─────────────────────────────────────────────────────────────────────────────
#  TOOLS — plain Python functions (NO @tool decorator needed in Lyzr)
#
#  KEY DIFFERENCE FROM LANGCHAIN:
#    LangChain: needs @tool + type hints to extract the schema
#    Lyzr:      plain functions — the SDK reads the docstring for description
# ─────────────────────────────────────────────────────────────────────────────

def search_knowledge_base(query: str) -> str:
    """
    Search the product knowledge base for help articles.
    Use for billing, API limits, SSO, password reset, or data export questions.
    Input: one or two keywords describing the topic.
    """
    q = query.lower()
    for key, article in KNOWLEDGE_BASE.items():
        if key in q or any(w in q for w in key.split("_")):
            return f"[KB: {key}] {article}"
    return "No article found. Direct user to docs.example.com."


def get_ticket_status(ticket_id: str) -> str:
    """
    Look up the status of an existing support ticket.
    Input: ticket ID string like 'TKT-001'.
    """
    tid = ticket_id.strip().upper()
    if tid in TICKET_DB:
        t = TICKET_DB[tid]
        return f"{tid}: {t['status']} | Issue: {t['issue']} | Age: {t['age']}"
    return f"{tid} not found. Verify the ticket number."


def create_ticket(issue: str, severity: str = "medium") -> str:
    """
    Create a new support ticket for the user.
    Use when KB doesn't resolve the issue.
    Args: issue (str), severity ('low'/'medium'/'high')
    """
    import random
    tid = f"TKT-{random.randint(100, 999)}"
    eta = {"low": "3-5 days", "medium": "24 hours", "high": "2 hours"}.get(severity.lower(), "24 hours")
    return f"Created {tid}. Issue: {issue}. Severity: {severity}. ETA: {eta}."


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD THE LYZR AGENT
#
#  COMPARE WITH LANGCHAIN (Phase 1):
#
#  LangChain:                         Lyzr ADK:
#  ─────────────────────────────────  ──────────────────────────────────────
#  llm = ChatOpenAI(model=...)        studio = Studio(api_key=...)
#  tools = [tool1, tool2]             # LLM set in create_agent(provider=...)
#  prompt = hub.pull("hwchase17/...")  # Prompt = instructions= field
#  agent = create_react_agent(...)    agent = studio.create_agent(...)
#  executor = AgentExecutor(...)      # No executor — Studio handles it
#  executor.invoke({"input": q})      agent.run(message=q)
# ─────────────────────────────────────────────────────────────────────────────

def build_lyzr_agent(studio: "Studio"):
    """
    Creates the tech support agent using Lyzr ADK.

    NOTE: Unlike LangChain where you manually wire the ReAct loop,
    Lyzr handles the orchestration internally. You only specify:
      - Which LLM to use
      - The agent's role/goal/instructions (system prompt)
      - Which tools to attach

    BUILT-IN EXTRAS you get for free in Lyzr:
      - Short-term memory (last N messages) — set memory=10
      - RAI guardrails (toxicity, PII, hallucination) — always on
      - Token tracking + latency in Studio dashboard — automatic
    """
    agent = studio.create_agent(
        name="TechSupportAgent-v1",
        provider="openai",           # LLM provider
        model="gpt-4o-mini",         # specific model
        role="Tech Support Specialist",
        goal="Resolve user support queries accurately and concisely",
        instructions=(
            "You are a tech support agent for a SaaS product. "
            "Always search the knowledge base first before creating tickets. "
            "If a KB article exists, cite the exact settings path. "
            "Only create a ticket if the KB cannot fully resolve the issue. "
            "Keep answers under 80 words."
        ),
        temperature=0.0,
        memory=10,                   # remember last 10 messages
    )

    # Attach tools — plain Python functions
    agent.add_tool(search_knowledge_base)
    agent.add_tool(get_ticket_status)
    agent.add_tool(create_ticket)

    return agent


# ─────────────────────────────────────────────────────────────────────────────
#  RUN THE AGENT
# ─────────────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "How do I update my billing information?",
    "What is the API rate limit for the Pro plan?",
    "I need to set up SSO with Okta for my enterprise account.",
    "What's the status of ticket TKT-003?",
    "My app keeps getting HTTP 429 errors, can you help?",
    "I haven't received my password reset email.",
    "How do I export my data to JSON format?",
]

def run_lyzr_agent(agent, queries: list[str]) -> list[dict]:
    """Run the Lyzr agent on all test queries."""
    results = []
    for i, query in enumerate(queries, 1):
        console.print(Rule(f"[bold cyan]Query {i}/{len(queries)}[/bold cyan]"))
        console.print(f"[bold blue]User:[/bold blue] {query}")
        try:
            response = agent.run(
                message=query,
                user_id="phase2-test-user",
                session_id=f"phase2-session-{i}",
            )
            # Lyzr response can be a string or object — normalise it
            output = response if isinstance(response, str) else str(response)
        except Exception as e:
            output = f"[ERROR] {type(e).__name__}: {e}"

        console.print(f"[bold green]Agent:[/bold green] {output[:200]}")
        results.append({"input": query, "output": output})

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  SIDE-BY-SIDE SDK COMPARISON (printed for learning)
# ─────────────────────────────────────────────────────────────────────────────

def print_sdk_comparison():
    """Print a side-by-side comparison of LangChain vs Lyzr ADK syntax."""
    console.print(Rule("[bold yellow]SDK Syntax Comparison: LangChain vs Lyzr[/bold yellow]"))

    table = Table(border_style="yellow", show_lines=True, title="How you build an agent")
    table.add_column("Concept", style="bold", width=20)
    table.add_column("LangChain (Phase 1)", width=38)
    table.add_column("Lyzr ADK (Phase 2)", width=38)

    rows = [
        ("Client / Entry point",
         "No single client;\nChatOpenAI + AgentExecutor",
         "Studio(api_key=...)\nOne object manages everything"),
        ("Define a tool",
         "@tool\ndef my_fn(x: str) -> str:\n  ...",
         "def my_fn(x: str) -> str:\n  ...\nagent.add_tool(my_fn)"),
        ("Set the LLM",
         "ChatOpenAI(model='gpt-4o-mini')",
         "studio.create_agent(\n  provider='openai',\n  model='gpt-4o-mini')"),
        ("Set the prompt",
         "hub.pull('hwchase17/react')\n+ PromptTemplate",
         "create_agent(\n  instructions='...',\n  role='...', goal='...')"),
        ("Add memory",
         "ConversationBufferMemory()\nor add to chain manually",
         "create_agent(memory=10)\n[built-in, just pass a number]"),
        ("Run the agent",
         "executor.invoke({'input': q})",
         "agent.run(message=q,\n  user_id=..., session_id=...)"),
        ("Get a trace ID",
         "run_id from callbacks\nor @traceable",
         "❌ Not returned by agent.run()"),
        ("Attach feedback",
         "client.create_feedback(\n  run_id, key, score)",
         "❌ No SDK method exists"),
        ("Guardrails",
         "Add manually as tools\nor LangGraph nodes",
         "✅ Built-in: toxicity,\nPII, hallucination, injection"),
    ]

    for row in rows:
        table.add_row(*row)

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Phase 2 — Step 1[/bold cyan]\n"
        "Same Agent, Built in Lyzr ADK\n\n"
        "[dim]We rebuild the Tech Support Agent using Lyzr's SDK\n"
        "and compare how the developer experience differs from LangChain.[/dim]",
        border_style="cyan"
    ))

    print_sdk_comparison()

    lyzr_key = os.getenv("LYZR_API_KEY")
    if not lyzr_key:
        console.print(Panel(
            "[red bold]LYZR_API_KEY not set.[/red bold]\n\n"
            "Add to your .env file:\n"
            "  LYZR_API_KEY=your-key-here\n\n"
            "Get it from: [link=https://studio.lyzr.ai]https://studio.lyzr.ai[/link] → Settings → API Keys\n\n"
            "[dim]The SDK comparison table above has been printed — it doesn't need the key.\n"
            "The live agent run below requires it.[/dim]",
            border_style="red",
            title="Missing API Key"
        ))
    else:
        console.print("\n[bold green]✓ LYZR_API_KEY found — running live agent...[/bold green]\n")
        studio = Studio(api_key=lyzr_key)
        agent  = build_lyzr_agent(studio)
        results = run_lyzr_agent(agent, TEST_QUERIES)

        console.print(Rule("[bold green]Run complete[/bold green]"))
        console.print(f"[dim]Ran {len(results)} queries via Lyzr ADK.[/dim]")
        console.print(
            "\n[bold]Now open [link=https://studio.lyzr.ai]studio.lyzr.ai[/link] "
            "→ Your agent → Traces tab[/bold]\n"
            "[dim]Notice: traces appear automatically — but you cannot query them "
            "programmatically from Python. That's the first gap.[/dim]"
        )

    console.print(
        "\n[bold yellow]Next:[/bold yellow] Run [bold]02_what_lyzr_has.py[/bold] "
        "to explore what Lyzr's APIs actually expose."
    )
