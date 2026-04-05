"""
Experiment runner and version comparison for the Lyzr EvalOps bridge.
Mirrors: langsmith.evaluation.evaluate() + compare_experiments().
"""

from .db import BridgeDB
from .evaluator import EVALUATORS


class ExperimentRunner:
    def __init__(self, db: BridgeDB = None):
        self.db = db or BridgeDB()

    def run(self, agent_fn, dataset_name: str,
            evaluator_names: list[str] = None,
            experiment_name: str = "experiment",
            agent_version: str = "v1",
            platform: str = "lyzr") -> str:
        """
        Run agent_fn on every example in the dataset,
        score each output with the specified evaluators,
        and store all results in SQLite.

        Returns: experiment_id
        """
        evaluator_names = evaluator_names or ["correctness", "helpfulness", "conciseness"]
        dataset = self.db.get_dataset_by_name(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        examples = self.db.list_examples(dataset["id"])
        exp_id   = self.db.create_experiment(
            name=experiment_name,
            dataset_id=dataset["id"],
            agent_version=agent_version,
            platform=platform,
            metadata={"evaluators": evaluator_names},
        )

        print(f"\n▶ Running experiment: {experiment_name}")
        print(f"  Dataset : {dataset_name} ({len(examples)} examples)")
        print(f"  Evaluators: {', '.join(evaluator_names)}\n")

        for i, ex in enumerate(examples, 1):
            input_text     = ex["inputs"].get("input", "")
            reference_text = ex["outputs"].get("output", "")

            print(f"  [{i}/{len(examples)}] {input_text[:55]}…")

            # Run the agent
            try:
                output_text = agent_fn(input_text)
            except Exception as e:
                output_text = f"[ERROR] {e}"

            # Score with each evaluator
            for ev_name in evaluator_names:
                ev_fn  = EVALUATORS.get(ev_name)
                if not ev_fn:
                    continue
                result = ev_fn(input_text, output_text, reference_text)
                self.db.save_score(
                    experiment_id=exp_id,
                    example_id=ex["id"],
                    input_text=input_text,
                    output_text=output_text,
                    evaluator=result["evaluator"],
                    score=result["score"],
                    comment=result["comment"],
                )

        print(f"\n  ✓ Experiment complete — id: {exp_id[:12]}…\n")
        return exp_id

    # ── Comparison + Release Decision ─────────────────────────────────────────

    def compare(self, exp_id_a: str, exp_id_b: str) -> dict:
        """
        Compare two experiments and return a structured release decision.

        Returns:
          {
            "exp_a": {...scores},  "exp_b": {...scores},
            "deltas": {...},
            "verdict": "PROMOTE" | "CANARY" | "HOLD" | "ROLLBACK",
            "reason": "..."
          }
        """
        scores_a = self.db.get_aggregate_scores(exp_id_a)
        scores_b = self.db.get_aggregate_scores(exp_id_b)
        metrics  = list(scores_a.keys())

        deltas       = {m: round(scores_b.get(m, 0) - scores_a.get(m, 0), 3) for m in metrics}
        improvements = sum(1 for d in deltas.values() if d >  0.03)
        regressions  = sum(1 for d in deltas.values() if d < -0.03)
        total        = len(metrics)

        if improvements == total:
            verdict = "PROMOTE"
            reason  = "v2 improves all metrics. Safe to promote to 100% traffic."
        elif improvements >= total - 1 and regressions == 0:
            verdict = "CANARY"
            reason  = (f"v2 improves {improvements}/{total} metrics with no regressions. "
                       "Roll out to 10% of traffic and monitor.")
        elif regressions >= improvements:
            verdict = "ROLLBACK"
            reason  = (f"v2 regresses on {regressions}/{total} metrics. "
                       "Do NOT promote. Revert if already deployed.")
        else:
            verdict = "HOLD"
            reason  = (f"Mixed results: {improvements} better, {regressions} worse. "
                       "Iterate before promoting.")

        return {
            "exp_a":   scores_a,
            "exp_b":   scores_b,
            "deltas":  deltas,
            "verdict": verdict,
            "reason":  reason,
        }
