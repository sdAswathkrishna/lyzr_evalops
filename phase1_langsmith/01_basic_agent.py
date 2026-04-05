"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 1 — Basic LangChain ReAct Agent (no tracing yet)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  - How to define custom tools using @tool
  - How a ReAct agent reasons (Thought → Action → Observation loop)
  - How to bind tools to an LLM and run an agent executor

THE AGENT:
  We are building a Tech Support Agent for a fictional SaaS product.
  It has three tools:
    1. search_knowledge_base  — looks up help articles
    2. get_ticket_status      — checks an existing support ticket
    3. create_ticket          — files a new support ticket

  This is representative of real agents teams build on platforms like Lyzr.

HOW TO RUN:
  python phase1_langsmith/01_basic_agent.py
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# ── LangChain imports ────────────────────────────────────────────────────────
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain import hub

load_dotenv()
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  MOCK DATA  (simulates a real knowledge base + ticketing system)
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

# ─────────────────────────────────────────────────────────────────────────────
#  TOOLS  — each @tool becomes an "action" the LLM can choose to call
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the product knowledge base for help articles.
    Use this when the user asks a question about product features,
    billing, API limits, SSO, or account settings.
    Input should be one or two keywords describing the topic.
    """
    query_lower = query.lower()
    # simple keyword match on mock KB
    for key, article in KNOWLEDGE_BASE.items():
        if key in query_lower or any(word in query_lower for word in key.split("_")):
            return f"[KB Article: {key}]\n{article}"
    return (
        "No direct article found. Suggest the user contact support "
        "or browse docs.example.com for more information."
    )


@tool
def get_ticket_status(ticket_id: str) -> str:
    """
    Look up the status of an existing support ticket.
    Use this when the user mentions a ticket number like TKT-001.
    Input must be the ticket ID (e.g. 'TKT-001').
    """
    ticket_id = ticket_id.strip().upper()
    if ticket_id in TICKET_DB:
        t = TICKET_DB[ticket_id]
        return (
            f"Ticket {ticket_id}: {t['status']}\n"
            f"Issue: {t['issue']}\n"
            f"Age: {t['age']}"
        )
    return f"Ticket {ticket_id} not found. Please verify the ticket number."


@tool
def create_ticket(issue: str, severity: str = "medium") -> str:
    """
    Create a new support ticket for the user's issue.
    Use this when the user has a problem that cannot be resolved
    by knowledge base articles alone.
    Inputs:
      - issue: a one-sentence description of the problem
      - severity: 'low', 'medium', or 'high'
    """
    import random
    new_id = f"TKT-{random.randint(100, 999)}"
    eta = {"low": "3-5 business days", "medium": "24 hours", "high": "2 hours"}.get(
        severity.lower(), "24 hours"
    )
    return (
        f"Ticket created: {new_id}\n"
        f"Issue: {issue}\n"
        f"Severity: {severity}\n"
        f"Expected response time: {eta}\n"
        f"You will receive a confirmation email shortly."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD THE AGENT
# ─────────────────────────────────────────────────────────────────────────────

def build_agent(model: str = "gpt-4o-mini") -> AgentExecutor:
    """
    Assembles the ReAct agent.

    CONCEPT — ReAct (Reasoning + Acting):
      The LLM alternates between:
        Thought:     "I need to check if there's a KB article about billing"
        Action:      search_knowledge_base("billing")
        Observation: [KB Article: billing] Billing is managed via...
        Thought:     "I have enough info to answer"
        Final Answer: "..."

      This loop continues until the LLM produces a Final Answer.
    """
    llm = ChatOpenAI(model=model, temperature=0)
    tools = [search_knowledge_base, get_ticket_status, create_ticket]

    # Pull the standard ReAct prompt from LangChain Hub
    # This prompt instructs the LLM to output in Thought/Action/Observation format
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,          # prints Thought/Action/Observation to console
        handle_parsing_errors=True,
        max_iterations=6,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TEST QUERIES  — these will become our evaluation dataset in later steps
# ─────────────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "How do I update my billing information?",
    "What is the API rate limit for the Pro plan?",
    "I need to set up SSO with Okta for my enterprise account.",
    "What's the status of ticket TKT-003?",
    "My app is getting HTTP 429 errors constantly, can you help?",
    "I haven't received my password reset email after 10 minutes.",
    "How do I export my data to JSON format?",
]


def run_agent_on_queries(agent: AgentExecutor, queries: list[str]) -> list[dict]:
    """Run the agent on a list of queries and collect input/output pairs."""
    results = []
    for i, query in enumerate(queries, 1):
        console.print(Rule(f"[bold cyan]Query {i}/{len(queries)}[/bold cyan]"))
        console.print(Panel(query, title="[bold]User Input[/bold]", border_style="blue"))

        try:
            result = agent.invoke({"input": query})
            output = result["output"]
        except Exception as e:
            output = f"ERROR: {e}"

        console.print(Panel(output, title="[bold green]Agent Answer[/bold green]", border_style="green"))
        results.append({"input": query, "output": output})

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Phase 1 — Step 1[/bold cyan]\n"
        "LangChain ReAct Agent  ·  No tracing yet\n\n"
        "[dim]The agent will reason through each query using the\n"
        "Thought → Action → Observation loop. Watch the verbose output.[/dim]",
        border_style="cyan"
    ))

    agent = build_agent(model="gpt-4o-mini")
    results = run_agent_on_queries(agent, TEST_QUERIES)

    console.print(Rule("[bold green]All queries complete[/bold green]"))
    console.print(f"\n[bold]Summary:[/bold] Ran [cyan]{len(results)}[/cyan] queries.")
    console.print(
        "\n[yellow]Next step:[/yellow] Run [bold]02_langsmith_tracing.py[/bold] "
        "to see these same runs captured in LangSmith with full trace visibility."
    )
