"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 4 — LLM-as-a-Judge Evaluations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  - What "LLM-as-a-judge" means and why it works
  - How to write evaluator functions that score agent outputs
  - The 3 types of evaluators: reference-based, criteria-based, pairwise
  - How evaluate() runs an agent against a dataset and stores scores
  - How to interpret evaluation results and make release decisions

CONCEPT — LLM-as-a-Judge:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Traditional testing: assert output == expected   ← too brittle │
  │                                                                  │
  │  LLM-as-a-Judge: use a separate LLM to score the answer         │
  │    Input:  (question, agent_answer, reference_answer)           │
  │    Output: { score: 0.0–1.0, reasoning: "..." }                 │
  │                                                                  │
  │  Works because: LLMs understand semantic similarity, tone,       │
  │  completeness — things string matching misses entirely           │
  └──────────────────────────────────────────────────────────────────┘

THREE EVALUATOR TYPES WE BUILD HERE:
  1. correctness_evaluator   — Does the answer match the reference?
  2. helpfulness_evaluator   — Is the answer helpful? (no reference needed)
  3. conciseness_evaluator   — Is the answer appropriately concise?

HOW TO RUN:
  python phase1_langsmith/04_run_evaluations.py

PREREQ: Run 03_create_dataset.py first.
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# ── LangSmith evaluation ─────────────────────────────────────────────────────
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example

# ── LangChain ────────────────────────────────────────────────────────────────
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain import hub
from langchain.evaluation import load_evaluator
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  RE-USE THE SAME TOOLS (no changes — we're evaluating the SAME agent)
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "billing":   "Billing is managed via the Settings → Billing page. "
                 "Invoices are sent on the 1st of each month. "
                 "To update your credit card, go to Settings → Payment Methods.",
    "api_limit": "Free tier: 100 API calls/day. Pro: 10,000/day. Enterprise: unlimited. "
                 "Rate limit errors return HTTP 429.",
    "reset":     "To reset your password, click 'Forgot Password' on the login page. "
                 "A reset link will be emailed within 2 minutes.",
    "sso":       "SSO is available on Enterprise plans. Providers: Okta, Azure AD, Google Workspace.",
    "export":    "Data export: Settings → Data → Export. Formats: CSV, JSON, Parquet.",
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
    q = query.lower()
    for key, article in KNOWLEDGE_BASE.items():
        if key in q or any(word in q for word in key.split("_")):
            return f"[KB Article: {key}]\n{article}"
    return "No article found. Direct user to docs.example.com."

@tool
def get_ticket_status(ticket_id: str) -> str:
    """Look up the status of a support ticket by ID (e.g. TKT-001)."""
    tid = ticket_id.strip().upper()
    if tid in TICKET_DB:
        t = TICKET_DB[tid]
        return f"Ticket {tid}: {t['status']}\nIssue: {t['issue']}\nAge: {t['age']}"
    return f"Ticket {tid} not found."

@tool
def create_ticket(issue: str, severity: str = "medium") -> str:
    """Create a new support ticket."""
    import random
    new_id = f"TKT-{random.randint(100, 999)}"
    eta = {"low": "3-5 days", "medium": "24 hours", "high": "2 hours"}.get(severity.lower(), "24 hours")
    return f"Ticket {new_id} created. Issue: {issue}. ETA: {eta}."


def build_agent_v1() -> AgentExecutor:
    """Agent v1 — standard gpt-4o-mini, no system prompt tuning."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_knowledge_base, get_ticket_status, create_ticket]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=False,
        handle_parsing_errors=True, max_iterations=6,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  THE AGENT RUNNER
#  evaluate() needs a callable: (inputs: dict) → (outputs: dict)
# ─────────────────────────────────────────────────────────────────────────────

def agent_runner(inputs: dict) -> dict:
    """
    The function evaluate() will call for each dataset example.
    Must accept 'inputs' dict and return 'outputs' dict.
    """
    agent = build_agent_v1()
    result = agent.invoke({"input": inputs["input"]})
    return {"output": result["output"]}


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATOR 1 — Correctness (reference-based)
#
#  CONCEPT: Compare agent answer to the reference (ground truth) answer.
#  The judge LLM scores how well the answer matches the reference.
#  Score: 1 = correct, 0 = incorrect
# ─────────────────────────────────────────────────────────────────────────────

def correctness_evaluator(run: Run, example: Example) -> dict:
    """
    LLM-as-a-judge: Is the agent's answer factually correct
    compared to the reference answer?

    This evaluator receives:
      run.outputs["output"]   → what the agent actually said
      example.outputs["output"] → what the correct answer is

    Returns: { "key": "correctness", "score": 0.0 or 1.0, "comment": "..." }
    """
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent_answer     = run.outputs.get("output", "") if run.outputs else ""
    reference_answer = example.outputs.get("output", "") if example.outputs else ""
    question         = example.inputs.get("input", "")

    prompt = f"""You are an expert evaluator for a tech support AI agent.

Question asked: {question}

Reference answer (ground truth):
{reference_answer}

Agent's answer:
{agent_answer}

Evaluate whether the agent's answer is factually CORRECT compared to the reference.
A correct answer must include all key facts from the reference.
Minor wording differences are fine as long as the facts are accurate.

Respond in exactly this format:
SCORE: [0 or 1]
REASON: [one sentence explanation]"""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    # Parse score from response
    score = 0.0
    for line in text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                score = 0.0

    reason = next(
        (l.replace("REASON:", "").strip() for l in text.split("\n") if l.startswith("REASON:")),
        text
    )

    return {"key": "correctness", "score": score, "comment": reason}


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATOR 2 — Helpfulness (criteria-based, no reference needed)
#
#  CONCEPT: Evaluate against abstract criteria rather than a specific answer.
#  Useful when there's no single "correct" answer — only better/worse ones.
# ─────────────────────────────────────────────────────────────────────────────

def helpfulness_evaluator(run: Run, example: Example) -> dict:
    """
    LLM-as-a-judge: Is the agent's answer genuinely helpful?
    Criteria: actionable, clear, doesn't leave the user confused.
    Score: 0.0 (unhelpful) to 1.0 (very helpful)
    """
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent_answer = run.outputs.get("output", "") if run.outputs else ""
    question     = example.inputs.get("input", "")

    prompt = f"""You are evaluating the helpfulness of a tech support AI agent's response.

User's question: {question}

Agent's response: {agent_answer}

Rate how HELPFUL this response is on a scale of 0.0 to 1.0:
  1.0 = Excellent: directly addresses the question, gives actionable next steps
  0.7 = Good: answers the question but could be clearer or more complete
  0.4 = Partial: partially answers but misses important information
  0.0 = Poor: doesn't help the user at all

Respond in exactly this format:
SCORE: [0.0 to 1.0]
REASON: [one sentence explanation]"""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    score = 0.5
    for line in text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                score = 0.5

    reason = next(
        (l.replace("REASON:", "").strip() for l in text.split("\n") if l.startswith("REASON:")),
        text
    )

    return {"key": "helpfulness", "score": score, "comment": reason}


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATOR 3 — Conciseness (criteria-based)
#
#  CONCEPT: Good support answers are complete but not verbose.
#  This evaluator catches agents that "over-explain" or pad their answers.
# ─────────────────────────────────────────────────────────────────────────────

def conciseness_evaluator(run: Run, example: Example) -> dict:
    """
    LLM-as-a-judge: Is the answer appropriately concise?
    Penalises overly long/padded responses and rewards clarity.
    """
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent_answer = run.outputs.get("output", "") if run.outputs else ""
    word_count   = len(agent_answer.split())

    prompt = f"""Evaluate the CONCISENESS of this tech support response.

Response ({word_count} words): {agent_answer}

A concise support response:
  - Answers the question without unnecessary padding
  - Does not repeat the question back
  - Avoids filler phrases like "Great question!" or "I hope this helps!"
  - Is complete but not verbose (ideal: 30–100 words for simple queries)

SCORE: [0.0 to 1.0]  (1.0 = perfectly concise, 0.0 = too verbose or too terse)
REASON: [one sentence]"""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    score = 0.5
    for line in text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                score = 0.5

    reason = next(
        (l.replace("REASON:", "").strip() for l in text.split("\n") if l.startswith("REASON:")),
        text
    )
    return {"key": "conciseness", "score": score, "comment": reason}


# ─────────────────────────────────────────────────────────────────────────────
#  RUN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(dataset_name: str, experiment_prefix: str = "v1") -> object:
    """
    Runs the agent against every example in the dataset,
    scores each output with all 3 evaluators, and stores results in LangSmith.

    evaluate() does this:
      for each example in dataset:
        output = agent_runner(example.inputs)
        for each evaluator:
          score  = evaluator(run=output_run, example=example)
          store score in LangSmith
    """
    console.print(Rule(f"[bold cyan]Running Evaluation: {experiment_prefix}[/bold cyan]"))
    console.print(f"[dim]Dataset: {dataset_name}[/dim]")
    console.print("[dim]Evaluators: correctness, helpfulness, conciseness[/dim]\n")

    results = evaluate(
        agent_runner,                          # the function being evaluated
        data=dataset_name,                     # dataset name or ID
        evaluators=[
            correctness_evaluator,
            helpfulness_evaluator,
            conciseness_evaluator,
        ],
        experiment_prefix=experiment_prefix,   # name shown in LangSmith UI
        metadata={
            "agent_version": "1.0.0",
            "model": "gpt-4o-mini",
            "phase": "phase1",
        },
        max_concurrency=2,                     # run 2 examples in parallel
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def display_results(results):
    """
    Parses the evaluate() results and prints a summary table.
    Then explains what the scores mean for a release decision.
    """
    console.print(Rule("[bold green]Evaluation Results[/bold green]"))

    # Aggregate scores across all examples
    # LangSmith >=0.2 returns an EvaluationResults object — iterate directly
    scores = {"correctness": [], "helpfulness": [], "conciseness": []}

    for result in results:
        # result may be a dict or an EvaluationResult object
        if isinstance(result, dict):
            eval_results = result.get("evaluation_results", {})
            for key in scores:
                if key in eval_results:
                    s = eval_results[key]
                    scores[key].append(s.score if hasattr(s, "score") else s)
        else:
            # Newer LangSmith: result has .feedback_results list
            feedback = getattr(result, "feedback_results", None) or []
            if not feedback:
                # Try alternate attribute names
                feedback = getattr(result, "evaluation_results", []) or []
            for fb in feedback:
                key = getattr(fb, "key", None)
                sc  = getattr(fb, "score", None)
                if key in scores and sc is not None:
                    scores[key].append(float(sc))

    # If still empty, try pulling scores from the results object directly
    if all(len(v) == 0 for v in scores.values()):
        try:
            df = results.to_pandas()
            for key in scores:
                col = f"feedback.{key}"
                if col in df.columns:
                    scores[key] = df[col].dropna().tolist()
        except Exception:
            pass

    # Summary table
    table = Table(title="Aggregate Evaluation Scores", border_style="green")
    table.add_column("Metric", style="bold")
    table.add_column("Avg Score", justify="center")
    table.add_column("Min", justify="center")
    table.add_column("Max", justify="center")
    table.add_column("Release Threshold", justify="center")
    table.add_column("Pass?", justify="center")

    thresholds = {"correctness": 0.8, "helpfulness": 0.7, "conciseness": 0.6}

    for metric, score_list in scores.items():
        if not score_list:
            continue
        avg = round(sum(score_list) / len(score_list), 2)
        mn  = round(min(score_list), 2)
        mx  = round(max(score_list), 2)
        threshold = thresholds[metric]
        passed = "[green]✓ PASS[/green]" if avg >= threshold else "[red]✗ FAIL[/red]"
        table.add_row(metric, str(avg), str(mn), str(mx), str(threshold), passed)

    console.print(table)

    # Release decision logic
    all_pass = all(
        (sum(s) / len(s) if s else 0) >= thresholds[k]
        for k, s in scores.items() if s
    )

    decision_color = "green" if all_pass else "red"
    decision_text  = "PROMOTE" if all_pass else "HOLD"
    decision_icon  = "🚀" if all_pass else "🚫"

    console.print(Panel(
        f"[bold {decision_color}]{decision_icon}  Release Decision: {decision_text}[/bold {decision_color}]\n\n"
        + (
            "All metrics are above threshold. Agent v1 is safe to promote.\n"
            "Next: run 05_compare_versions.py to validate against a v2 candidate."
            if all_pass else
            "One or more metrics are below threshold. Agent should NOT be promoted.\n"
            "Investigate failing examples in LangSmith and improve the agent first."
        ),
        border_style=decision_color,
        title="[bold]EvalOps Decision Gate[/bold]"
    ))

    console.print(
        "\n[bold]View full results in LangSmith:[/bold]\n"
        "  Projects → lyzr-evalops-phase1 → Experiments tab\n"
        "  Click the experiment to see per-example scores and LLM judge reasoning."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATASET_NAME = "tech-support-agent-v1"

    console.print(Panel.fit(
        "[bold cyan]Phase 1 — Step 4[/bold cyan]\n"
        "LLM-as-a-Judge Evaluations\n\n"
        "[dim]3 evaluators will score every example in the dataset:\n"
        "  • Correctness  — matches reference answer?\n"
        "  • Helpfulness  — actually useful to the user?\n"
        "  • Conciseness  — clear and not verbose?\n\n"
        "A release decision gate will recommend: PROMOTE or HOLD[/dim]",
        border_style="cyan"
    ))

    results = run_evaluation(DATASET_NAME, experiment_prefix="agent-v1")
    display_results(results)

    console.print(
        "\n[bold yellow]Next:[/bold yellow] Run [bold]05_compare_versions.py[/bold] "
        "to build a v2 agent and compare it against v1 side-by-side."
    )
