"""
SQLite persistence layer for the Lyzr EvalOps bridge.

Tables:
  traces       — every intercepted agent.run() call
  feedback     — scores attached to traces
  datasets     — named collections of eval examples
  examples     — individual (input, reference_output) pairs
  experiments  — a named run of (agent × dataset × evaluators)
  eval_scores  — per-example scores from an experiment
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).parent.parent / "poc_data.db"


class BridgeDB:
    def __init__(self, db_path: str = None):
        self.path = db_path or str(DB_PATH)
        self._init_schema()
        self.seed_default_prompts()
        self.seed_builtin_evaluators()

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS traces (
                    id          TEXT PRIMARY KEY,
                    project     TEXT,
                    platform    TEXT DEFAULT 'lyzr',
                    input       TEXT,
                    output      TEXT,
                    latency_ms  REAL,
                    token_count INTEGER,
                    status      TEXT,
                    user_id     TEXT,
                    session_id  TEXT,
                    agent_version TEXT DEFAULT 'v1',
                    timestamp   TEXT
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id          TEXT PRIMARY KEY,
                    trace_id    TEXT,
                    key         TEXT,
                    score       REAL,
                    comment     TEXT,
                    timestamp   TEXT
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    id          TEXT PRIMARY KEY,
                    name        TEXT UNIQUE,
                    description TEXT,
                    created_at  TEXT
                );

                CREATE TABLE IF NOT EXISTS examples (
                    id              TEXT PRIMARY KEY,
                    dataset_id      TEXT,
                    inputs_json     TEXT,
                    outputs_json    TEXT,
                    metadata_json   TEXT,
                    source_trace_id TEXT,
                    created_at      TEXT
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    id            TEXT PRIMARY KEY,
                    name          TEXT,
                    dataset_id    TEXT,
                    agent_version TEXT,
                    platform      TEXT DEFAULT 'lyzr',
                    metadata_json TEXT,
                    created_at    TEXT
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    id            TEXT PRIMARY KEY,
                    name          TEXT,
                    dataset_id    TEXT,
                    agent_version TEXT,
                    platform      TEXT DEFAULT 'lyzr',
                    metadata_json TEXT,
                    created_at    TEXT
                );

                CREATE TABLE IF NOT EXISTS eval_scores (
                    id             TEXT PRIMARY KEY,
                    evaluation_id  TEXT,
                    example_id     TEXT,
                    input_text     TEXT,
                    output_text    TEXT,
                    evaluator      TEXT,
                    model          TEXT DEFAULT 'gpt-4o-mini',
                    score          REAL,
                    comment        TEXT,
                    created_at     TEXT
                );

                CREATE TABLE IF NOT EXISTS evaluator_definitions (
                    id           TEXT PRIMARY KEY,
                    name         TEXT UNIQUE NOT NULL,
                    description  TEXT NOT NULL,
                    judge_prompt TEXT NOT NULL,
                    is_builtin   INTEGER DEFAULT 0,
                    created_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prompts (
                    id          TEXT PRIMARY KEY,
                    version     TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    is_active   INTEGER DEFAULT 0,
                    created_at  TEXT NOT NULL,
                    notes       TEXT DEFAULT ''
                );
            """)

    # ── Traces ────────────────────────────────────────────────────────────────

    def save_trace(self, trace: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO traces
                (id, project, platform, input, output, latency_ms,
                 token_count, status, user_id, session_id, agent_version, timestamp)
                VALUES (:id,:project,:platform,:input,:output,:latency_ms,
                        :token_count,:status,:user_id,:session_id,:agent_version,:timestamp)
            """, trace)

    def list_traces(self, project: str = None, platform: str = None,
                    agent_version: str = None, limit: int = 100):
        q = "SELECT * FROM traces WHERE 1=1"
        params = []
        if project:
            q += " AND project = ?"; params.append(project)
        if platform:
            q += " AND platform = ?"; params.append(platform)
        if agent_version:
            q += " AND agent_version = ?"; params.append(agent_version)
        q += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    def get_trace(self, trace_id: str):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM traces WHERE id=?", (trace_id,)).fetchone()
            return dict(row) if row else None

    # ── Feedback ──────────────────────────────────────────────────────────────

    def save_feedback(self, trace_id: str, key: str, score: float, comment: str = ""):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO feedback (id, trace_id, key, score, comment, timestamp)
                VALUES (?,?,?,?,?,?)
            """, (str(uuid.uuid4()), trace_id, key, score, comment,
                  datetime.utcnow().isoformat()))

    def get_feedback(self, trace_id: str):
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM feedback WHERE trace_id=?", (trace_id,)).fetchall()]

    # ── Datasets ──────────────────────────────────────────────────────────────

    def create_dataset(self, name: str, description: str = "") -> str:
        dataset_id = str(uuid.uuid4())
        with self._conn() as conn:
            # Delete if exists (for re-runs)
            existing = conn.execute(
                "SELECT id FROM datasets WHERE name=?", (name,)).fetchone()
            if existing:
                conn.execute("DELETE FROM examples WHERE dataset_id=?", (existing["id"],))
                conn.execute("DELETE FROM datasets WHERE name=?", (name,))
            conn.execute("""
                INSERT INTO datasets (id, name, description, created_at)
                VALUES (?,?,?,?)
            """, (dataset_id, name, description, datetime.utcnow().isoformat()))
        return dataset_id

    def get_dataset_by_name(self, name: str):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE name=?", (name,)).fetchone()
            return dict(row) if row else None

    def list_datasets(self):
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM datasets ORDER BY created_at DESC").fetchall()]

    # ── Examples ──────────────────────────────────────────────────────────────

    def add_example(self, dataset_id: str, inputs: dict, outputs: dict,
                    metadata: dict = None, source_trace_id: str = None) -> str:
        ex_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO examples
                (id, dataset_id, inputs_json, outputs_json, metadata_json,
                 source_trace_id, created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (ex_id, dataset_id, json.dumps(inputs), json.dumps(outputs),
                  json.dumps(metadata or {}), source_trace_id,
                  datetime.utcnow().isoformat()))
        return ex_id

    def list_examples(self, dataset_id: str):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM examples WHERE dataset_id=? ORDER BY created_at",
                (dataset_id,)).fetchall()
            return [{**dict(r),
                     "inputs": json.loads(r["inputs_json"]),
                     "outputs": json.loads(r["outputs_json"]),
                     "metadata": json.loads(r["metadata_json"])}
                    for r in rows]

    # ── Evaluations (renamed from Experiments) ────────────────────────────────

    def create_evaluation(self, name: str, dataset_id: str,
                          agent_version: str = "v1", platform: str = "lyzr",
                          metadata: dict = None) -> str:
        eval_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO evaluations
                (id, name, dataset_id, agent_version, platform, metadata_json, created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (eval_id, name, dataset_id, agent_version, platform,
                  json.dumps(metadata or {}), datetime.utcnow().isoformat()))
        return eval_id

    # Backward-compat alias used by run_poc.py
    def create_experiment(self, name: str, dataset_id: str,
                          agent_version: str = "v1", platform: str = "lyzr",
                          metadata: dict = None) -> str:
        return self.create_evaluation(name, dataset_id, agent_version, platform, metadata)

    def list_evaluations(self):
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM evaluations ORDER BY created_at DESC").fetchall()]

    def list_experiments(self):
        return self.list_evaluations()

    def get_evaluation(self, eval_id: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM evaluations WHERE id=?", (eval_id,)).fetchone()
            return dict(row) if row else None

    def get_experiment(self, exp_id: str):
        return self.get_evaluation(exp_id)

    # ── Eval Scores ───────────────────────────────────────────────────────────

    def save_score(self, evaluation_id: str, example_id: str,
                   input_text: str, output_text: str,
                   evaluator: str, score: float, comment: str = "",
                   model: str = "gpt-4o-mini"):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO eval_scores
                (id, evaluation_id, example_id, input_text, output_text,
                 evaluator, model, score, comment, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (str(uuid.uuid4()), evaluation_id, example_id,
                  input_text, output_text, evaluator, model, score, comment,
                  datetime.utcnow().isoformat()))

    def get_scores(self, evaluation_id: str):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_scores WHERE evaluation_id=?",
                (evaluation_id,)).fetchall()
            # Fallback: try old column name experiment_id for legacy rows
            if not rows:
                rows = conn.execute(
                    "SELECT * FROM eval_scores WHERE experiment_id=?",
                    (evaluation_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_aggregate_scores(self, evaluation_id: str) -> dict:
        scores = self.get_scores(evaluation_id)
        result = {}
        for s in scores:
            ev = s["evaluator"]
            result.setdefault(ev, []).append(s["score"])
        return {ev: round(sum(v)/len(v), 3) for ev, v in result.items()}

    def get_scores_by_model(self, evaluation_id: str) -> dict:
        """
        Returns scores structured for the detail table view:
        { example_id: { model: { evaluator: {score, comment} } } }
        """
        rows = self.get_scores(evaluation_id)
        result = {}
        for s in rows:
            eid = s["example_id"]
            mdl = s.get("model", "gpt-4o-mini")
            ev  = s["evaluator"]
            result.setdefault(eid, {}).setdefault(mdl, {})[ev] = {
                "score":   s["score"],
                "comment": s.get("comment", ""),
                "input":   s.get("input_text", ""),
                "output":  s.get("output_text", ""),
            }
        return result

    # ── Evaluator Definitions ─────────────────────────────────────────────────

    _BUILTIN_EVALUATORS = [
        {
            "name": "ACCURACY",
            "description": "Score: pass/fail (pass=accurate, fail=inaccurate). Evaluate whether the response contains correct information. Check if the facts align with the given context or knowledge.",
            "judge_prompt": "Question: {input}\nAgent response: {output}\nReference: {reference}\n\nDoes the response contain accurate, correct information compared to the reference?\nSCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]",
        },
        {
            "name": "CLARIFY",
            "description": "Evaluate whether the response asks clarifying questions only when truly needed to understand or proceed, instead of turning the chat into an interview.",
            "judge_prompt": "Question: {input}\nAgent response: {output}\n\nDoes the response ask clarifying questions only when truly necessary?\nSCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]",
        },
        {
            "name": "LOWER_FRICTION",
            "description": "Evaluate whether the response cuts user effort by quickly acknowledging any frustration.",
            "judge_prompt": "Question: {input}\nAgent response: {output}\n\nDoes the response reduce user effort and acknowledge frustration appropriately?\nSCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]",
        },
        {
            "name": "UNCLEAR_INTENT",
            "description": "Evaluate whether the response correctly understands and addresses the user's true intent, even when their phrasing is vague, indirect, or poorly worded.",
            "judge_prompt": "Question: {input}\nAgent response: {output}\n\nDoes the response correctly address the user's true intent despite vague phrasing?\nSCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]",
        },
        {
            "name": "CONVERSATIONAL_LANGUAGE",
            "description": "Checks whether the response uses natural, conversational language instead of overly formal or technical wording.",
            "judge_prompt": "Agent response: {output}\n\nDoes the response use natural, conversational language rather than overly formal or technical wording?\nSCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]",
        },
        {
            "name": "PERSONA_WE_VOICE",
            "description": "Checks whether the response uses first-person plural ('we/our') instead of detached third-person phrasing when referring to the product or company.",
            "judge_prompt": "Agent response: {output}\n\nDoes the response use 'we/our' when referring to the product or company instead of third-person?\nSCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]",
        },
        {
            "name": "FLOW_DIRECTNESS",
            "description": "Evaluate if the response directs the user to the next possible step.",
            "judge_prompt": "Question: {input}\nAgent response: {output}\n\nDoes the response clearly direct the user toward a next actionable step?\nSCORE: [1 for PASS, 0 for FAIL]\nREASON: [one sentence]",
        },
    ]

    def seed_builtin_evaluators(self):
        with self._conn() as conn:
            for ev in self._BUILTIN_EVALUATORS:
                existing = conn.execute(
                    "SELECT id FROM evaluator_definitions WHERE name=?", (ev["name"],)
                ).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO evaluator_definitions "
                        "(id, name, description, judge_prompt, is_builtin, created_at) "
                        "VALUES (?,?,?,?,?,?)",
                        (str(uuid.uuid4()), ev["name"], ev["description"],
                         ev["judge_prompt"], 1, datetime.utcnow().isoformat())
                    )

    def list_evaluator_definitions(self):
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM evaluator_definitions ORDER BY is_builtin DESC, name ASC"
            ).fetchall()]

    def create_evaluator_definition(self, name: str, description: str,
                                    judge_prompt: str) -> str:
        ev_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO evaluator_definitions "
                "(id, name, description, judge_prompt, is_builtin, created_at) "
                "VALUES (?,?,?,?,0,?)",
                (ev_id, name, description, judge_prompt, datetime.utcnow().isoformat())
            )
        return ev_id

    def get_evaluator_definition(self, ev_id: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM evaluator_definitions WHERE id=?", (ev_id,)
            ).fetchone()
            return dict(row) if row else None

    # ── Trace → Dataset workflow ───────────────────────────────────────────────

    def add_trace_to_dataset(self, trace_id: str, dataset_id: str) -> str | None:
        """
        Promote a production trace into a dataset example.

        This is the key EvalOps workflow:
          chat message → trace saved → human reviews trace → clicks "Add to Dataset"
          → trace becomes a labeled example → dataset grows over time
          → next evaluation run uses real production queries

        The trace's input becomes the example 'input'.
        The trace's output becomes the 'reference_output' (the ground truth we're
        evaluating future agent versions against).

        Returns the new example_id, or None if trace not found.
        """
        trace = self.get_trace(trace_id)
        if not trace:
            return None

        # Check this trace isn't already in this dataset
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM examples WHERE source_trace_id=? AND dataset_id=?",
                (trace_id, dataset_id)
            ).fetchone()
            if existing:
                return existing["id"]   # idempotent — return existing example id

        ex_id = self.add_example(
            dataset_id=dataset_id,
            inputs={"question": trace["input"]},
            outputs={"answer": trace["output"]},
            metadata={
                "source": "production_trace",
                "agent_version": trace.get("agent_version", "unknown"),
                "platform": trace.get("platform", "unknown"),
                "latency_ms": trace.get("latency_ms"),
                "promoted_from_trace": trace_id,
            },
            source_trace_id=trace_id,
        )
        return ex_id

    def get_or_create_dataset(self, name: str, description: str = "") -> dict:
        """Return existing dataset by name, or create it if it doesn't exist."""
        existing = self.get_dataset_by_name(name)
        if existing:
            return existing
        dataset_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO datasets (id, name, description, created_at)
                VALUES (?,?,?,?)
            """, (dataset_id, name, description, datetime.utcnow().isoformat()))
        return {"id": dataset_id, "name": name, "description": description}

    def list_traces_in_dataset(self, dataset_id: str) -> list:
        """Return all trace IDs already added to a dataset (to avoid duplicates in UI)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT source_trace_id FROM examples WHERE dataset_id=? AND source_trace_id IS NOT NULL",
                (dataset_id,)
            ).fetchall()
            return [r["source_trace_id"] for r in rows]

    # ── Prompts ───────────────────────────────────────────────────────────────

    _DEFAULT_PROMPTS = [
        {
            "version": "v1",
            "name": "Initial Prompt",
            "content": (
                "You are a tech support agent for ACME SaaS platform. "
                "You have a knowledge base with articles about billing, API limits, SSO, "
                "password reset, and data export. You also have a ticketing system. "
                "Search the knowledge base first. Only create a ticket if the KB "
                "cannot fully resolve the issue. Be helpful and accurate."
            ),
            "notes": "Original baseline prompt.",
            "is_active": 1,
        },
        {
            "version": "v2",
            "name": "Concise + Structured",
            "content": (
                "You are a concise tech support agent for ACME SaaS platform. "
                "RULES: "
                "1. Always cite the exact settings path (e.g. Settings → Billing). "
                "2. Search KB first; only create a ticket if KB doesn't fully resolve it. "
                "3. Keep answers under 80 words. "
                "4. Never use filler phrases like 'Great question!'. "
                "5. If a KB article exists, always mention the specific navigation path."
            ),
            "notes": "Added navigation paths, word limit, tone rules.",
            "is_active": 0,
        },
    ]

    def seed_default_prompts(self):
        """Insert v1 and v2 prompts if the table is empty."""
        with self._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
            if count == 0:
                for p in self._DEFAULT_PROMPTS:
                    conn.execute(
                        "INSERT INTO prompts (id, version, name, content, is_active, created_at, notes) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (str(uuid.uuid4()), p["version"], p["name"],
                         p["content"], p["is_active"],
                         datetime.utcnow().isoformat(), p["notes"])
                    )

    def list_prompts(self) -> list:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM prompts ORDER BY created_at ASC").fetchall()]

    def get_prompt(self, prompt_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM prompts WHERE id=?", (prompt_id,)).fetchone()
            return dict(row) if row else None

    def get_active_prompt(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM prompts WHERE is_active=1 ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def activate_prompt(self, prompt_id: str) -> bool:
        with self._conn() as conn:
            conn.execute("UPDATE prompts SET is_active=0")
            result = conn.execute(
                "UPDATE prompts SET is_active=1 WHERE id=?", (prompt_id,)
            )
            return result.rowcount > 0

    def create_prompt(self, version: str, name: str, content: str,
                      notes: str = "", set_active: bool = False) -> str:
        prompt_id = str(uuid.uuid4())
        with self._conn() as conn:
            if set_active:
                conn.execute("UPDATE prompts SET is_active=0")
            conn.execute(
                "INSERT INTO prompts (id, version, name, content, is_active, created_at, notes) "
                "VALUES (?,?,?,?,?,?,?)",
                (prompt_id, version, name, content,
                 1 if set_active else 0,
                 datetime.utcnow().isoformat(), notes)
            )
        return prompt_id
