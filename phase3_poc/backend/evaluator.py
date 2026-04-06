"""
LLM-as-a-Judge evaluators — Sparrow-style names.
Supports running each evaluator across multiple models for side-by-side comparison.

Built-in evaluators: ACCURACY, CLARIFY, LOWER_FRICTION, UNCLEAR_INTENT,
                     CONVERSATIONAL_LANGUAGE, PERSONA_WE_VOICE, FLOW_DIRECTNESS

Dynamic evaluators: loaded from evaluator_definitions table and run with the
                    same judge pattern using the stored judge_prompt template.
"""

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

EVAL_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]


def _call_judge(prompt: str, model: str = "gpt-4o-mini") -> tuple[float, str]:
    """Call the judge LLM and parse SCORE / REASON from the response."""
    llm  = ChatOpenAI(model=model, temperature=0)
    text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    score  = 0.5
    reason = text
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("SCORE:"):
            try:
                val = stripped.replace("SCORE:", "").strip().split()[0]
                score = float(val)
            except (ValueError, IndexError):
                pass
        if stripped.startswith("REASON:"):
            reason = stripped.replace("REASON:", "").strip()
    return score, reason


def _run_evaluator(judge_prompt_template: str, input_text: str,
                   output_text: str, reference: str = "",
                   model: str = "gpt-4o-mini") -> dict:
    """Run a single evaluator with a filled prompt template."""
    prompt = judge_prompt_template.format(
        input=input_text,
        output=output_text,
        reference=reference,
    )
    score, reason = _call_judge(prompt, model=model)
    return {"score": score, "comment": reason}


def run_all_models(judge_prompt_template: str, evaluator_name: str,
                   input_text: str, output_text: str,
                   reference: str = "",
                   models: list = None) -> list[dict]:
    """
    Run one evaluator across all specified models.

    Returns list of:
      { evaluator, model, score, comment }
    """
    models = models or EVAL_MODELS
    results = []
    for model in models:
        try:
            r = _run_evaluator(judge_prompt_template, input_text, output_text, reference, model)
            results.append({
                "evaluator": evaluator_name,
                "model":     model,
                "score":     r["score"],
                "comment":   r["comment"],
            })
        except Exception as e:
            results.append({
                "evaluator": evaluator_name,
                "model":     model,
                "score":     0.0,
                "comment":   f"Error: {e}",
            })
    return results


# ── Built-in judge prompt templates ──────────────────────────────────────────

_JUDGE_PROMPTS = {
    "ACCURACY": (
        "Question: {input}\nAgent response: {output}\nReference: {reference}\n\n"
        "Does the response contain accurate, correct information compared to the reference?\n"
        "SCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]"
    ),
    "CLARIFY": (
        "Question: {input}\nAgent response: {output}\n\n"
        "Does the response ask clarifying questions only when truly necessary?\n"
        "SCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]"
    ),
    "LOWER_FRICTION": (
        "Question: {input}\nAgent response: {output}\n\n"
        "Does the response reduce user effort and acknowledge frustration appropriately?\n"
        "SCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]"
    ),
    "UNCLEAR_INTENT": (
        "Question: {input}\nAgent response: {output}\n\n"
        "Does the response correctly address the user's true intent despite vague phrasing?\n"
        "SCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]"
    ),
    "CONVERSATIONAL_LANGUAGE": (
        "Agent response: {output}\n\n"
        "Does the response use natural, conversational language rather than overly formal wording?\n"
        "SCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]"
    ),
    "PERSONA_WE_VOICE": (
        "Agent response: {output}\n\n"
        "Does the response use 'we/our' when referring to the product or company?\n"
        "SCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]"
    ),
    "FLOW_DIRECTNESS": (
        "Question: {input}\nAgent response: {output}\n\n"
        "Does the response clearly direct the user toward a next actionable step?\n"
        "SCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]"
    ),
}


def get_judge_prompt(evaluator_name: str, db=None) -> str | None:
    """
    Look up a judge prompt by evaluator name.
    Checks built-ins first, then DB-defined evaluators.
    """
    if evaluator_name in _JUDGE_PROMPTS:
        return _JUDGE_PROMPTS[evaluator_name]
    if db:
        defs = db.list_evaluator_definitions()
        for d in defs:
            if d["name"] == evaluator_name:
                return d["judge_prompt"]
    return None


# ── Simple function-based API (used by existing run_poc.py) ──────────────────
# Wraps the new system so old call sites keep working.

def correctness(input_text: str, output_text: str, reference: str = "") -> dict:
    score, reason = _call_judge(_JUDGE_PROMPTS["ACCURACY"].format(
        input=input_text, output=output_text, reference=reference))
    return {"evaluator": "ACCURACY", "score": score, "comment": reason}


def helpfulness(input_text: str, output_text: str, reference: str = "") -> dict:
    score, reason = _call_judge(_JUDGE_PROMPTS["LOWER_FRICTION"].format(
        input=input_text, output=output_text, reference=reference))
    return {"evaluator": "LOWER_FRICTION", "score": score, "comment": reason}


def conciseness(input_text: str, output_text: str, reference: str = "") -> dict:
    score, reason = _call_judge(_JUDGE_PROMPTS["FLOW_DIRECTNESS"].format(
        input=input_text, output=output_text, reference=reference))
    return {"evaluator": "FLOW_DIRECTNESS", "score": score, "comment": reason}


EVALUATORS = {
    "ACCURACY":                correctness,
    "LOWER_FRICTION":          helpfulness,
    "FLOW_DIRECTNESS":         conciseness,
    # legacy keys so old code doesn't break
    "correctness":             correctness,
    "helpfulness":             helpfulness,
    "conciseness":             conciseness,
}
