"""
Lyzr agent using the sync lyzr_agent_api REST client,
wrapped with the EvalOps bridge tracer.
"""

import os, time
from dotenv import load_dotenv
load_dotenv()

from lyzr_agent_api.client import AgentAPI
from lyzr_agent_api.models.environment import EnvironmentConfig, FeatureConfig
from lyzr_agent_api.models.agents import AgentConfig
from lyzr_agent_api.models.chat import ChatRequest

from .tools import TEST_QUERIES, search_knowledge_base, get_ticket_status, create_ticket
from ..bridge.tracer import LyzrTracer
from ..bridge.db import BridgeDB

LYZR_API_KEY = os.getenv("LYZR_API_KEY")
BASE_URL      = "https://agent.api.lyzr.app"

# ─────────────────────────────────────────────────────────────────────────────
#  System prompts per version
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "v1": (
        "You are a tech support agent for a SaaS product. "
        "You have a knowledge base with articles about billing, API limits, SSO, "
        "password reset, and data export. You also have a ticketing tool. "
        "Search the knowledge base first. Only create a ticket if the KB "
        "cannot fully resolve the issue. Be helpful and accurate."
    ),
    "v2": (
        "You are a concise tech support agent for a SaaS product. "
        "RULES: "
        "1. Always cite the exact settings path (e.g. Settings → Billing). "
        "2. Search KB first; only create a ticket if KB doesn't fully resolve it. "
        "3. Keep answers under 80 words. "
        "4. Never use filler phrases like 'Great question!'. "
        "5. If KB article exists, always mention the specific navigation path."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  A simple in-process agent that uses the tools directly
#  (since the lyzr_agent_api doesn't support custom Python tool execution,
#   we implement the same logic with direct tool calls + LLM)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleLyzrAgent:
    """
    Lyzr-flavoured agent that uses lyzr_agent_api for the LLM call
    and calls tools in-process (matching what Lyzr Studio does internally).
    This is what we intercept with the bridge tracer.
    """

    def __init__(self, version: str = "v1"):
        self.version       = version
        self.system_prompt = SYSTEM_PROMPTS[version]
        self.client        = AgentAPI(
            x_api_key=LYZR_API_KEY,
            base_url=BASE_URL,
        )
        self._env_id   = None
        self._agent_id = None
        self._setup()

    def _setup(self):
        """Create environment + agent via the REST API."""
        try:
            env_cfg = EnvironmentConfig(
                name=f"tech-support-env-{self.version}",
                features=[FeatureConfig(type="SHORT_TERM_MEMORY", config={})],
                tools=[],
                llm_config={
                    "provider": "openai",
                    "model":    "gpt-4o-mini",
                    "config":   {"temperature": 0},
                    "env": {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "")},
                },
            )
            env_resp = self.client.create_environment_endpoint(json_body=env_cfg)
            self._env_id = env_resp.get("env_id") or env_resp.get("id")

            agent_cfg = AgentConfig(
                env_id=self._env_id,
                system_prompt=self.system_prompt,
                name=f"TechSupportAgent-{self.version}",
                agent_description="Tech support agent for SaaS product",
            )
            agent_resp = self.client.create_agent_endpoint(json_body=agent_cfg)
            self._agent_id = agent_resp.get("agent_id") or agent_resp.get("id")

        except Exception as e:
            # Fallback: use OpenAI directly with the same system prompt
            self._env_id   = None
            self._agent_id = None
            self._fallback_reason = str(e)

    def run(self, message: str, user_id: str = "anon",
            session_id: str = None, **kwargs) -> str:
        """Run one turn of the agent."""

        # If Lyzr API setup succeeded, use it
        if self._agent_id:
            try:
                req = ChatRequest(
                    user_id=user_id,
                    agent_id=self._agent_id,
                    message=self._build_message(message),
                    session_id=session_id or user_id,
                )
                resp = self.client.chat_with_agent(json_body=req)
                if isinstance(resp, dict):
                    return resp.get("response") or resp.get("message") or str(resp)
                return str(resp)
            except Exception as e:
                pass  # fall through to local fallback

        # Fallback: run locally with OpenAI + tool logic (same model, same prompt)
        return self._local_run(message)

    def _build_message(self, user_msg: str) -> str:
        """
        Augment user message with tool results so the LLM has context.
        This mirrors what Lyzr's internal tool-calling does.
        """
        kb_result     = search_knowledge_base(user_msg)
        ticket_hint   = ""
        # check if message mentions a ticket ID
        import re
        m = re.search(r'TKT-\d+', user_msg, re.IGNORECASE)
        if m:
            ticket_hint = f"\n[Ticket lookup]: {get_ticket_status(m.group())}"

        return (
            f"User question: {user_msg}\n\n"
            f"[Knowledge base search result]: {kb_result}"
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
            HumanMessage(content=self._build_message(message)),
        ]
        return llm.invoke(msgs).content


# ─────────────────────────────────────────────────────────────────────────────
#  Build traced agent
# ─────────────────────────────────────────────────────────────────────────────

def build_traced_agent(version: str = "v1", db: BridgeDB = None) -> LyzrTracer:
    agent  = SimpleLyzrAgent(version=version)
    return LyzrTracer(
        agent=agent,
        project="lyzr-evalops-poc",
        agent_version=version,
        db=db,
    )


def run_all(version: str = "v1", queries=None, db: BridgeDB = None):
    tracer  = build_traced_agent(version=version, db=db)
    queries = queries or TEST_QUERIES
    results = []
    for query, user_id in queries:
        output, trace_id = tracer.run(
            message=query, user_id=user_id,
            session_id=f"poc-{version}-{user_id}"
        )
        results.append({"input": query, "output": output,
                         "trace_id": trace_id, "user_id": user_id})
        time.sleep(0.3)
    return results, tracer
