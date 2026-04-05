"""
LLM-as-a-Judge evaluators for the Lyzr EvalOps bridge.
Three evaluators: correctness, helpfulness, conciseness.
Judge model: gpt-4o-mini (same as Phase 1 for fair comparison).
"""

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


JUDGE_MODEL = "gpt-4o-mini"


def _call_judge(prompt: str) -> tuple[float, str]:
    """Call the judge LLM and parse SCORE / REASON from the response."""
    llm  = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    score = 0.5
    reason = text
    for line in text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                pass
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
    return score, reason


# ── Evaluator 1 — Correctness ─────────────────────────────────────────────────

def correctness(input_text: str, output_text: str, reference: str) -> dict:
    """
    Does the agent answer match the reference (ground truth)?
    Score: 1.0 = correct, 0.0 = incorrect.
    """
    prompt = (
        f"Question: {input_text}\n"
        f"Reference answer: {reference}\n"
        f"Agent answer: {output_text}\n\n"
        "Is the agent's answer factually correct compared to the reference? "
        "Minor wording differences are fine.\n"
        "SCORE: [0 or 1]\nREASON: [one sentence]"
    )
    score, reason = _call_judge(prompt)
    return {"evaluator": "correctness", "score": score, "comment": reason}


# ── Evaluator 2 — Helpfulness ─────────────────────────────────────────────────

def helpfulness(input_text: str, output_text: str, reference: str = "") -> dict:
    """
    Is the answer genuinely useful to the user?
    Score: 0.0 (useless) to 1.0 (excellent).
    """
    prompt = (
        f"Question: {input_text}\n"
        f"Agent answer: {output_text}\n\n"
        "Rate helpfulness 0.0-1.0. "
        "1.0 = directly answers with actionable steps. "
        "0.0 = doesn't help at all.\n"
        "SCORE: [0.0-1.0]\nREASON: [one sentence]"
    )
    score, reason = _call_judge(prompt)
    return {"evaluator": "helpfulness", "score": score, "comment": reason}


# ── Evaluator 3 — Conciseness ─────────────────────────────────────────────────

def conciseness(input_text: str, output_text: str, reference: str = "") -> dict:
    """
    Is the answer appropriately concise?
    Score: 0.0 (very verbose) to 1.0 (perfectly concise).
    """
    words  = len(output_text.split())
    prompt = (
        f"Response ({words} words): {output_text}\n\n"
        "Rate conciseness 0.0-1.0. "
        "1.0 = complete and brief (30-100 words). "
        "0.0 = too verbose or too terse.\n"
        "SCORE: [0.0-1.0]\nREASON: [one sentence]"
    )
    score, reason = _call_judge(prompt)
    return {"evaluator": "conciseness", "score": score, "comment": reason}


# ── Registry ──────────────────────────────────────────────────────────────────

EVALUATORS = {
    "correctness": correctness,
    "helpfulness":  helpfulness,
    "conciseness":  conciseness,
}
