"""
agent.py — Lyzr ADK agent (equivalent capability to the LangChain agent).

This agent implements the same ACME tech-support use case using the Lyzr ADK
(lyzr_agent_api REST client) instead of LangChain, allowing side-by-side
comparison of the two frameworks' EvalOps capabilities.

Framework comparison:
  LangChain agent: ReAct loop + @tool decorators + FAISS retrieval
  Lyzr ADK agent:  lyzr_agent_api REST client + in-process tool calls + same KB

Both agents are wrapped with the same BridgeDB tracer so traces appear in
the same dashboard and can be compared directly.
"""

import os, sys, time, re
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# File lives at: phase3_poc/agents/lyzr_adk_agent/agent.py
# parents[3] = lyzr_comparison/ ← ROOT
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Lyzr ADK imports ──────────────────────────────────────────────────────────
from lyzr_agent_api.client import AgentAPI
from lyzr_agent_api.models.environment import EnvironmentConfig, FeatureConfig
from lyzr_agent_api.models.agents import AgentConfig
from lyzr_agent_api.models.chat import ChatRequest

# ── Backend imports ───────────────────────────────────────────────────────────
from phase3_poc.backend.tracer import LyzrTracer
from phase3_poc.backend.db import BridgeDB

# ── Shared knowledge base tools (reuse LangChain agent's tool functions) ──────
# We import the raw Python functions (without @tool decorator side-effects)
# to reuse the same FAISS KB and ticket DB across both agent implementations.
from phase3_poc.agents.langchain_agent.agent import (
    TICKET_DB,
    _load_vectorstore,
)

LYZR_API_KEY = os.getenv("LYZR_API_KEY")
BASE_URL      = "https://agent.api.lyzr.app"

# ── Support test queries (same as LangChain agent for fair comparison) ────────
TEST_QUERIES = [
    ("How do I update my billing information?",          "user_1"),
    ("What is the API rate limit for the Growth plan?",  "user_2"),
    ("I need to set up SSO with Okta.",                  "user_3"),
    ("What's the status of ticket TKT-003?",             "user_4"),
    ("My app keeps getting HTTP 429 errors.",            "user_5"),
    ("I haven't received my password reset email.",      "user_6"),
    ("How do I export my data to JSON format?",          "user_7"),
]


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS — versioned for prompt A/B comparison
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPTS = {
    "v1": (
        "You are a tech support agent for ACME SaaS platform. "
        "You have a knowledge base with articles about billing, API limits, SSO, "
        "password reset, and data export. You also have a ticketing system. "
        "Search the knowledge base first. Only create a ticket if the KB "
        "cannot fully resolve the issue. Be helpful and accurate."
    ),
    "v2": (
        "You are a concise tech support agent for ACME SaaS platform. "
        "RULES: "
        "1. Always cite the exact settings path (e.g. Settings → Billing). "
        "2. Search KB first; only create a ticket if KB doesn't fully resolve it. "
        "3. Keep answers under 80 words. "
        "4. Never use filler phrases like 'Great question!'. "
        "5. If a KB article exists, always mention the specific navigation path."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL HELPERS (plain Python — not @tool decorated, since Lyzr ADK calls them
# in-process rather than through LangChain's tool routing)
# ══════════════════════════════════════════════════════════════════════════════

def _search_knowledge_base(query: str) -> str:
    """FAISS semantic search — same index as the LangChain agent."""
    try:
        vs      = _load_vectorstore()
        results = vs.similarity_search_with_score(query, k=3)
        if not results:
            return "No relevant documentation found."
        lines = []
        for doc, score in results:
            source  = Path(doc.metadata.get("source", "?")).name.replace(".txt", "")
            quality = "strong" if score < 0.9 else "moderate" if score < 1.2 else "weak"
            lines.append(f"[{source} — {quality}] {doc.page_content.strip()}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Knowledge base unavailable: {e}"


def _get_ticket_status(ticket_id: str) -> str:
    """Ticket status lookup against the shared TICKET_DB."""
    tid = ticket_id.strip().upper()
    if tid not in TICKET_DB:
        return f"Ticket {tid} not found."
    t = TICKET_DB[tid]
    return (f"{tid}: {t['status'].upper()} | {t['issue']} | "
            f"Priority: {t['priority'].upper()} | User: {t['user']}")


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE LYZR AGENT
# ══════════════════════════════════════════════════════════════════════════════

class SimpleLyzrAgent:
    """
    Lyzr-flavoured agent using lyzr_agent_api REST client for the LLM call.
    Tools are called in-process (same pattern Lyzr Studio uses internally).

    Falls back to direct OpenAI call if the Lyzr API is unavailable, ensuring
    the demo always produces output.
    """

    def __init__(self, version: str = "v1"):
        self.version       = version
        self.system_prompt = SYSTEM_PROMPTS.get(version, SYSTEM_PROMPTS["v1"])
        self.client        = AgentAPI(x_api_key=LYZR_API_KEY, base_url=BASE_URL)
        self._env_id       = None
        self._agent_id     = None
        self._setup()

    def _setup(self):
        """Create Lyzr environment + agent via REST API."""
        try:
            env_cfg = EnvironmentConfig(
                name=f"acme-support-env-{self.version}",
                features=[FeatureConfig(type="SHORT_TERM_MEMORY", config={})],
                tools=[],
                llm_config={
                    "provider": "openai",
                    "model":    "gpt-4o-mini",
                    "config":   {"temperature": 0},
                    "env":      {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "")},
                },
            )
            env_resp       = self.client.create_environment_endpoint(json_body=env_cfg)
            self._env_id   = env_resp.get("env_id") or env_resp.get("id")

            agent_cfg = AgentConfig(
                env_id=self._env_id,
                system_prompt=self.system_prompt,
                name=f"AcmeSupportAgent-{self.version}",
                agent_description="ACME SaaS customer support agent",
            )
            agent_resp     = self.client.create_agent_endpoint(json_body=agent_cfg)
            self._agent_id = agent_resp.get("agent_id") or agent_resp.get("id")

        except Exception as e:
            # Fallback: will use local OpenAI call in run()
            self._env_id   = None
            self._agent_id = None

    def run(self, message: str, user_id: str = "anon",
            session_id: str = None, **kwargs) -> str:
        """Run one conversation turn."""
        if self._agent_id:
            try:
                req  = ChatRequest(
                    user_id=user_id,
                    agent_id=self._agent_id,
                    message=self._augment_message(message),
                    session_id=session_id or user_id,
                )
                resp = self.client.chat_with_agent(json_body=req)
                if isinstance(resp, dict):
                    return resp.get("response") or resp.get("message") or str(resp)
                return str(resp)
            except Exception:
                pass

        return self._local_run(message)

    def _augment_message(self, user_msg: str) -> str:
        """
        Augment the user message with tool results before sending to the LLM.
        This is how Lyzr's internal tool-calling works: retrieve context first,
        then pass it all to the LLM in a single prompt.
        """
        kb_result   = _search_knowledge_base(user_msg)
        ticket_hint = ""
        m = re.search(r"TKT-\d+", user_msg, re.IGNORECASE)
        if m:
            ticket_hint = f"\n[Ticket lookup]: {_get_ticket_status(m.group())}"

        return (
            f"User question: {user_msg}\n\n"
            f"[Knowledge base context]:\n{kb_result}"
            f"{ticket_hint}\n\n"
            "Please answer the user question using the above context."
        )

    def _local_run(self, message: str) -> str:
        """Direct OpenAI call — used when Lyzr API is unavailable."""
        from langchain_openai import ChatOpenAI
        from langchain.schema import SystemMessage, HumanMessage
        llm  = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        msgs = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._augment_message(message)),
        ]
        return llm.invoke(msgs).content


# ══════════════════════════════════════════════════════════════════════════════
# BUILD TRACED AGENT
# ══════════════════════════════════════════════════════════════════════════════

def build_traced_agent(version: str = "v1", db: BridgeDB = None) -> LyzrTracer:
    """Build a LyzrTracer-wrapped SimpleLyzrAgent."""
    agent = SimpleLyzrAgent(version=version)
    return LyzrTracer(
        agent=agent,
        project="lyzr-evalops-poc",
        agent_version=version,
        db=db,
    )


def run_all(version: str = "v1", queries=None, db: BridgeDB = None):
    """Run all test queries through the traced agent. Returns (results, tracer)."""
    tracer  = build_traced_agent(version=version, db=db)
    queries = queries or TEST_QUERIES
    results = []
    for query, user_id in queries:
        output, trace_id = tracer.run(
            message=query, user_id=user_id,
            session_id=f"poc-{version}-{user_id}"
        )
        results.append({
            "input":    query,
            "output":   output,
            "trace_id": trace_id,
            "user_id":  user_id,
        })
        time.sleep(0.3)
    return results, tracer


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  Lyzr ADK Agent — Standalone Test")
    print("=" * 65)

    agent = SimpleLyzrAgent(version="v1")
    for q in ["How do I reset my password?",
               "What's the Growth plan price?",
               "Status of TKT-003?"]:
        print(f"\nQ: {q}")
        print(f"A: {agent.run(q, user_id='test')}")
