"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 5 — Compare Agent Versions (v1 vs v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  - How to create a v2 agent with different behaviour (better system prompt)
  - How to run both versions against the SAME dataset
  - How to compare experiments side-by-side in LangSmith
  - How to make a PROMOTE / CANARY / HOLD / ROLLBACK decision
  - The release decision framework (the core of what you're building for Lyzr)

THE v1 vs v2 DIFFERENCE:
  v1: standard ReAct agent, generic prompt, no special instructions
  v2: ReAct agent + a system prompt that tells it to:
        - Always mention the specific settings path
        - Always suggest creating a ticket if KB article doesn't fully help
        - Keep answers under 100 words

  This models a REAL release scenario: the team improved the system prompt
  and wants to know if v2 is actually better before promoting it to production.

RELEASE DECISION FRAMEWORK:
  ┌────────────────────────────────────────────────────────────────────┐
  │  PROMOTE   → v2 beats v1 on all key metrics by ≥5%                │
  │  CANARY    → v2 beats v1 on most metrics but not all (gradual roll)│
  │  HOLD      → v2 is mixed — some metrics better, some worse         │
  │  ROLLBACK  → v2 is worse than v1 across the board                 │
  └────────────────────────────────────────────────────────────────────┘

HOW TO RUN:
  python phase1_langsmith/05_compare_versions.py

PREREQ: Run 03_create_dataset.py first.
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from langsmith import Client
from langsmith.evaluation import evaluate

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain import hub
from langchain.schema import HumanMessage
from langsmith.schemas import Run, Example

load_dotenv()
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED TOOLS
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "billing":   "Billing is managed via Settings → Billing. "
                 "Invoices on the 1st of each month. Credit card: Settings → Payment Methods.",
    "api_limit": "Free: 100/day. Pro: 10,000/day. Enterprise: unlimited. "
                 "429 = rate limit exceeded. Increase via Settings → API → Request Increase.",
    "reset":     "Click 'Forgot Password' on login. Email arrives in 2 min. Check spam.",
    "sso":       "SSO on Enterprise only. Providers: Okta, Azure AD, Google Workspace. "
                 "Guide: docs.example.com/sso",
    "export":    "Settings → Data → Export. Formats: CSV, JSON, Parquet. Async, emailed.",
}
TICKET_DB = {
    "TKT-001": {"status": "Open",     "issue": "API returning 500 errors",    "age": "2 hours"},
    "TKT-002": {"status": "Resolved", "issue": "Billing overcharge",          "age": "3 days"},
    "TKT-003": {"status": "Pending",  "issue": "SSO configuration help",      "age": "1 day"},
    "TKT-004": {"status": "Open",     "issue": "Password reset not received", "age": "30 mins"},
}

@tool
def search_knowledge_base(query: str) -> str:
    """Search the product knowledge base for help articles."""
    q = query.lower()
    for key, article in KNOWLEDGE_BASE.items():
        if key in q or any(w in q for w in key.split("_")):
            return f"[KB: {key}] {article}"
    return "No article found. Direct user to docs.example.com."

@tool
def get_ticket_status(ticket_id: str) -> str:
    """Look up a support ticket status by ID (e.g. TKT-001)."""
    tid = ticket_id.strip().upper()
    if tid in TICKET_DB:
        t = TICKET_DB[tid]
        return f"{tid}: {t['status']} — {t['issue']} ({t['age']} ago)"
    return f"{tid} not found."

@tool
def create_ticket(issue: str, severity: str = "medium") -> str:
    """Create a support ticket. severity: low / medium / high."""
    import random
    tid = f"TKT-{random.randint(100, 999)}"
    eta = {"low": "3-5 days", "medium": "24 hours", "high": "2 hours"}.get(severity.lower(), "24h")
    return f"Created {tid}: {issue} (severity={severity}, ETA={eta})"

TOOLS = [search_knowledge_base, get_ticket_status, create_ticket]


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT v1  — baseline (same as steps 1–4)
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_v1() -> AgentExecutor:
    """
    v1: Standard ReAct agent.
    No special instructions — just the default hwchase17/react prompt.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent, tools=TOOLS, verbose=False,
        handle_parsing_errors=True, max_iterations=6,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT v2  — improved version
#
#  THE CHANGE: A system-level instruction is injected into the prompt.
#  This is the most common type of "agent improvement" in practice.
#  Teams tweak system prompts, swap models, change tools — then re-evaluate.
# ─────────────────────────────────────────────────────────────────────────────

V2_SYSTEM_INSTRUCTIONS = """You are a concise, expert tech support agent for SaaS products.

RULES:
1. Always include the exact settings path when directing users (e.g. "Settings → Billing")
2. If a KB article exists, cite it first, then add your own context
3. If the issue can't be fully resolved with the KB, ALWAYS offer to create a ticket
4. Keep answers under 80 words — be direct, not verbose
5. Never say "Great question!" or use filler phrases
6. If a ticket exists, quote its current status explicitly
"""

def build_agent_v2() -> AgentExecutor:
    """
    v2: ReAct agent with improved system instructions.
    Changes vs v1:
      - Stricter formatting rules (settings paths, word limit)
      - Always suggests ticket creation as fallback
      - No filler phrases
    """
    # Inject the v2 system instructions via a custom prompt
    from langchain.prompts import PromptTemplate

    # We build a custom ReAct prompt that includes our system instructions
    react_template = V2_SYSTEM_INSTRUCTIONS + """

Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(react_template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent, tools=TOOLS, verbose=False,
        handle_parsing_errors=True, max_iterations=6,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  RUNNER FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def make_runner(build_fn):
    """Returns a runner function that evaluate() can call."""
    def runner(inputs: dict) -> dict:
        agent = build_fn()
        result = agent.invoke({"input": inputs["input"]})
        return {"output": result["output"]}
    return runner

agent_v1_runner = make_runner(build_agent_v1)
agent_v2_runner = make_runner(build_agent_v2)


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED EVALUATORS (same as step 4 — key for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

def correctness_evaluator(run: Run, example: Example) -> dict:
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    answer = run.outputs.get("output", "") if run.outputs else ""
    reference = example.outputs.get("output", "") if example.outputs else ""
    question = example.inputs.get("input", "")
    prompt = (
        f"Question: {question}\nReference: {reference}\nAgent: {answer}\n\n"
        "Is the agent's answer factually correct vs the reference?\n"
        "SCORE: [0 or 1]\nREASON: [one sentence]"
    )
    response = judge.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    score = 0.0
    for line in text.split("\n"):
        if line.startswith("SCORE:"):
            try: score = float(line.replace("SCORE:", "").strip())
            except: pass
    reason = next((l.replace("REASON:", "").strip() for l in text.split("\n") if l.startswith("REASON:")), text)
    return {"key": "correctness", "score": score, "comment": reason}

def helpfulness_evaluator(run: Run, example: Example) -> dict:
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    answer = run.outputs.get("output", "") if run.outputs else ""
    question = example.inputs.get("input", "")
    prompt = (
        f"Question: {question}\nAgent: {answer}\n\n"
        "Rate helpfulness 0.0-1.0 (1.0=excellent, 0.0=useless)\n"
        "SCORE: [0.0 to 1.0]\nREASON: [one sentence]"
    )
    response = judge.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    score = 0.5
    for line in text.split("\n"):
        if line.startswith("SCORE:"):
            try: score = float(line.replace("SCORE:", "").strip())
            except: pass
    reason = next((l.replace("REASON:", "").strip() for l in text.split("\n") if l.startswith("REASON:")), text)
    return {"key": "helpfulness", "score": score, "comment": reason}

def conciseness_evaluator(run: Run, example: Example) -> dict:
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    answer = run.outputs.get("output", "") if run.outputs else ""
    words = len(answer.split())
    prompt = (
        f"Response ({words} words): {answer}\n\n"
        "Rate conciseness 0.0-1.0 (1.0=perfectly concise, 0.0=very verbose)\n"
        "SCORE: [0.0 to 1.0]\nREASON: [one sentence]"
    )
    response = judge.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    score = 0.5
    for line in text.split("\n"):
        if line.startswith("SCORE:"):
            try: score = float(line.replace("SCORE:", "").strip())
            except: pass
    reason = next((l.replace("REASON:", "").strip() for l in text.split("\n") if l.startswith("REASON:")), text)
    return {"key": "conciseness", "score": score, "comment": reason}

EVALUATORS = [correctness_evaluator, helpfulness_evaluator, conciseness_evaluator]


# ─────────────────────────────────────────────────────────────────────────────
#  RUN BOTH EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

def run_both_experiments(dataset_name: str):
    """Evaluate v1 and v2 against the same dataset, store in LangSmith."""
    console.print(Rule("[bold cyan]Running Experiments: v1 and v2[/bold cyan]"))
    console.print("[dim]Both agents will be evaluated against the same dataset[/dim]")
    console.print("[dim]with the same 3 evaluators for a fair comparison.\n[/dim]")

    console.print("[bold yellow]▶ Running v1 evaluation...[/bold yellow]")
    results_v1 = evaluate(
        agent_v1_runner,
        data=dataset_name,
        evaluators=EVALUATORS,
        experiment_prefix="agent-v1",
        metadata={"version": "1.0.0", "model": "gpt-4o-mini", "prompt": "standard-react"},
        max_concurrency=2,
    )

    console.print("\n[bold yellow]▶ Running v2 evaluation...[/bold yellow]")
    results_v2 = evaluate(
        agent_v2_runner,
        data=dataset_name,
        evaluators=EVALUATORS,
        experiment_prefix="agent-v2",
        metadata={"version": "2.0.0", "model": "gpt-4o-mini", "prompt": "improved-system-prompt"},
        max_concurrency=2,
    )

    return results_v1, results_v2


# ─────────────────────────────────────────────────────────────────────────────
#  AGGREGATE SCORES
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results) -> dict:
    """Compute average score per metric from evaluate() results."""
    scores = {"correctness": [], "helpfulness": [], "conciseness": []}

    for result in results:
        if isinstance(result, dict):
            eval_results = result.get("evaluation_results", {})
            for key in scores:
                if key in eval_results:
                    s = eval_results[key]
                    scores[key].append(s.score if hasattr(s, "score") else s)
        else:
            feedback = getattr(result, "feedback_results", None) or []
            if not feedback:
                feedback = getattr(result, "evaluation_results", []) or []
            for fb in feedback:
                key = getattr(fb, "key", None)
                sc  = getattr(fb, "score", None)
                if key in scores and sc is not None:
                    scores[key].append(float(sc))

    # Fallback: pull from pandas DataFrame
    if all(len(v) == 0 for v in scores.values()):
        try:
            df = results.to_pandas()
            for key in scores:
                col = f"feedback.{key}"
                if col in df.columns:
                    scores[key] = df[col].dropna().tolist()
        except Exception:
            pass

    return {
        k: round(sum(v) / len(v), 3) if v else 0.0
        for k, v in scores.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
#  RELEASE DECISION ENGINE
#
#  THIS IS THE CORE OF WHAT YOU'RE BUILDING FOR LYZR.
#  The logic here maps to: Promote / Canary / Hold / Rollback
# ─────────────────────────────────────────────────────────────────────────────

def make_release_decision(v1_scores: dict, v2_scores: dict) -> tuple[str, str]:
    """
    Compares v1 and v2 scores and returns a release decision.

    Decision logic:
      PROMOTE  → v2 improves ALL metrics by ≥ 5%
      CANARY   → v2 improves MOST metrics (≥2/3) with no major regressions
      HOLD     → mixed results — some better, some worse
      ROLLBACK → v2 is worse than v1 on most metrics

    Returns: (decision, explanation)
    """
    metrics = list(v1_scores.keys())
    improvements = 0
    regressions  = 0
    deltas = {}

    for m in metrics:
        delta = v2_scores[m] - v1_scores[m]
        deltas[m] = delta
        if delta > 0.03:        improvements += 1
        elif delta < -0.03:     regressions  += 1

    total = len(metrics)

    if improvements == total:
        return "PROMOTE", "v2 improves all metrics. Safe to promote to production."
    elif improvements >= total - 1 and regressions == 0:
        return "CANARY", (
            f"v2 improves {improvements}/{total} metrics with no regressions. "
            "Roll out gradually (canary 10% → 50% → 100%)."
        )
    elif regressions == 0:
        return "HOLD", (
            "v2 shows marginal changes. Not enough improvement to justify a release. "
            "Investigate and iterate before promoting."
        )
    elif regressions >= improvements:
        return "ROLLBACK", (
            f"v2 regresses on {regressions}/{total} metrics. "
            "Do NOT promote. Investigate what changed and revert if already deployed."
        )
    else:
        return "HOLD", (
            f"Mixed results: {improvements} improvements, {regressions} regressions. "
            "Dig into the failing examples before promoting."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def display_comparison(v1_scores: dict, v2_scores: dict, decision: str, explanation: str):
    """Print the side-by-side comparison table and release decision."""
    console.print(Rule("[bold cyan]Version Comparison Results[/bold cyan]"))

    # Side-by-side table
    table = Table(title="Agent v1 vs Agent v2", border_style="cyan", show_lines=True)
    table.add_column("Metric",     style="bold", width=16)
    table.add_column("v1 Score",   justify="center", width=12)
    table.add_column("v2 Score",   justify="center", width=12)
    table.add_column("Δ Delta",    justify="center", width=10)
    table.add_column("Trend",      justify="center", width=8)

    for metric in v1_scores:
        v1  = v1_scores[metric]
        v2  = v2_scores[metric]
        delta = round(v2 - v1, 3)
        if delta > 0.03:
            trend = "[green]↑ Better[/green]"
            delta_str = f"[green]+{delta}[/green]"
        elif delta < -0.03:
            trend = "[red]↓ Worse[/red]"
            delta_str = f"[red]{delta}[/red]"
        else:
            trend = "[yellow]→ Same[/yellow]"
            delta_str = f"[yellow]{delta}[/yellow]"
        table.add_row(metric, str(v1), str(v2), delta_str, trend)

    console.print(table)

    # Release decision banner
    colors = {
        "PROMOTE":  "green",
        "CANARY":   "yellow",
        "HOLD":     "yellow",
        "ROLLBACK": "red",
    }
    icons = {
        "PROMOTE":  "🚀",
        "CANARY":   "🐤",
        "HOLD":     "⏸️ ",
        "ROLLBACK": "🔄",
    }
    color = colors.get(decision, "white")
    icon  = icons.get(decision, "")

    console.print(Panel(
        f"[bold {color}]{icon}  Release Decision: {decision}[/bold {color}]\n\n"
        f"{explanation}\n\n"
        "[dim]─────────────────────────────────────────────────────────\n"
        "In production, this decision would trigger:\n"
        "  PROMOTE  → deploy v2 to 100% of traffic\n"
        "  CANARY   → deploy v2 to 10% of traffic, monitor, then ramp\n"
        "  HOLD     → keep v1 running, iterate on v2 offline\n"
        "  ROLLBACK → revert to v1 immediately, alert the team[/dim]",
        border_style=color,
        title="[bold]EvalOps Release Decision[/bold]"
    ))

    # LangSmith UI instructions
    console.print(Panel(
        "[bold green]Compare in LangSmith UI:[/bold green]\n\n"
        "  1. Go to Datasets & Testing → tech-support-agent-v1\n"
        "  2. Click 'Compare Experiments'\n"
        "  3. Select 'agent-v1-...' and 'agent-v2-...'\n"
        "  4. You'll see a side-by-side diff of every example\n"
        "  5. Click any row to see exactly where v2 improved or regressed\n\n"
        "[dim]This is the key LangSmith feature we'll replicate in Lyzr.[/dim]",
        border_style="blue"
    ))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATASET_NAME = "tech-support-agent-v1"

    console.print(Panel.fit(
        "[bold cyan]Phase 1 — Step 5[/bold cyan]\n"
        "Agent v1 vs v2 — Version Comparison & Release Decision\n\n"
        "[dim]v1: Standard ReAct agent\n"
        "v2: Same agent + improved system prompt (concise, always cite paths)\n\n"
        "Same dataset. Same evaluators. Objective comparison.[/dim]",
        border_style="cyan"
    ))

    results_v1, results_v2 = run_both_experiments(DATASET_NAME)

    v1_scores = aggregate(results_v1)
    v2_scores = aggregate(results_v2)

    decision, explanation = make_release_decision(v1_scores, v2_scores)
    display_comparison(v1_scores, v2_scores, decision, explanation)

    console.print(Panel(
        "[bold green]Phase 1 Complete![/bold green]\n\n"
        "You have now built and experienced:\n"
        "  ✓ LangChain ReAct agent with custom tools\n"
        "  ✓ LangSmith tracing (traces, spans, metadata, feedback)\n"
        "  ✓ Trace → Dataset pipeline\n"
        "  ✓ LLM-as-a-judge evaluation (3 custom evaluators)\n"
        "  ✓ Version comparison with a structured release decision\n\n"
        "[bold yellow]Next:[/bold yellow] Phase 2 — Audit Lyzr's existing infrastructure\n"
        "and map it against what LangSmith provides.",
        border_style="green",
        title="[bold]Phase 1 Summary[/bold]"
    ))
