"""
Runtime interceptor for Lyzr agents.

Wraps agent.run() transparently — the agent never knows it's being traced.
Captures: input, output, latency, timestamp, user_id, session_id, agent_version.
Stores everything in the local SQLite BridgeDB.
"""

import uuid
import time
from datetime import datetime
from .db import BridgeDB


class LyzrTracer:
    """
    Drop-in wrapper around any Lyzr agent object.

    Usage:
        agent   = studio.create_agent(...)
        tracer  = LyzrTracer(agent, project="my-project", agent_version="v1")
        output, trace_id = tracer.run("How do I reset my password?", user_id="u1")

    The original agent is unchanged. All traces go to SQLite.
    """

    def __init__(self, agent, project: str = "lyzr-evalops-poc",
                 agent_version: str = "v1", db: BridgeDB = None):
        self._agent        = agent
        self.project       = project
        self.agent_version = agent_version
        self.db            = db or BridgeDB()
        self._traces       = []           # in-memory cache for current session

    # ── Core intercept ────────────────────────────────────────────────────────

    def run(self, message: str, user_id: str = "anon",
            session_id: str = None, **kwargs) -> tuple[str, str]:
        """
        Run the agent and capture a trace.

        Returns: (output_string, trace_id)
        """
        trace_id   = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        start_ms   = time.time() * 1000

        try:
            response   = self._agent.run(
                message=message, user_id=user_id,
                session_id=session_id, **kwargs
            )
            # Lyzr may return a string or an object with .message
            if hasattr(response, "message"):
                output = response.message
            elif hasattr(response, "content"):
                output = response.content
            else:
                output = str(response)
            status = "success"
        except Exception as exc:
            output = f"[ERROR] {type(exc).__name__}: {exc}"
            status = "error"

        latency_ms = round(time.time() * 1000 - start_ms, 1)

        trace = {
            "id":            trace_id,
            "project":       self.project,
            "platform":      "lyzr",
            "input":         message,
            "output":        output,
            "latency_ms":    latency_ms,
            "token_count":   self._estimate_tokens(message, output),
            "status":        status,
            "user_id":       user_id,
            "session_id":    session_id,
            "agent_version": self.agent_version,
            "timestamp":     datetime.utcnow().isoformat(),
        }

        self.db.save_trace(trace)
        self._traces.append(trace)
        return output, trace_id

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _estimate_tokens(self, inp: str, out: str) -> int:
        """Rough 1-token-per-4-chars estimate when exact counts unavailable."""
        return (len(inp) + len(out)) // 4

    def add_feedback(self, trace_id: str, key: str, score: float, comment: str = ""):
        """Attach a named score to a previously captured trace."""
        self.db.save_feedback(trace_id, key, score, comment)

    @property
    def session_traces(self) -> list[dict]:
        """All traces captured in this session (in-memory)."""
        return self._traces

    def last_trace_id(self) -> str | None:
        return self._traces[-1]["id"] if self._traces else None
