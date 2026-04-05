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

                CREATE TABLE IF NOT EXISTS eval_scores (
                    id            TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    example_id    TEXT,
                    input_text    TEXT,
                    output_text   TEXT,
                    evaluator     TEXT,
                    score         REAL,
                    comment       TEXT,
                    created_at    TEXT
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

    # ── Experiments ───────────────────────────────────────────────────────────

    def create_experiment(self, name: str, dataset_id: str,
                          agent_version: str = "v1", platform: str = "lyzr",
                          metadata: dict = None) -> str:
        exp_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO experiments
                (id, name, dataset_id, agent_version, platform, metadata_json, created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (exp_id, name, dataset_id, agent_version, platform,
                  json.dumps(metadata or {}), datetime.utcnow().isoformat()))
        return exp_id

    def list_experiments(self):
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC").fetchall()]

    def get_experiment(self, exp_id: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
            return dict(row) if row else None

    # ── Eval Scores ───────────────────────────────────────────────────────────

    def save_score(self, experiment_id: str, example_id: str,
                   input_text: str, output_text: str,
                   evaluator: str, score: float, comment: str = ""):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO eval_scores
                (id, experiment_id, example_id, input_text, output_text,
                 evaluator, score, comment, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (str(uuid.uuid4()), experiment_id, example_id,
                  input_text, output_text, evaluator, score, comment,
                  datetime.utcnow().isoformat()))

    def get_scores(self, experiment_id: str):
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM eval_scores WHERE experiment_id=?",
                (experiment_id,)).fetchall()]

    def get_aggregate_scores(self, experiment_id: str) -> dict:
        scores = self.get_scores(experiment_id)
        result = {}
        for s in scores:
            ev = s["evaluator"]
            result.setdefault(ev, []).append(s["score"])
        return {ev: round(sum(v)/len(v), 3) for ev, v in result.items()}

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
