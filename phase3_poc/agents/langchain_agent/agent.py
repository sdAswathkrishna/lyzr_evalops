"""
agent.py — LangChain ReAct Agent with FAISS embeddings and LangSmith tracing.

HOW THE AGENT WORKS:
=====================

  User message
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │  ReAct Loop  (Reasoning + Acting)                   │
  │                                                     │
  │  THOUGHT: "I need to look up SSO config steps"      │
  │     │                                               │
  │     ▼                                               │
  │  ACTION: search_knowledge_base("SSO config Okta")   │
  │     │                                               │
  │     ▼                                               │
  │  OBSERVATION: [top-4 chunks from FAISS returned]    │
  │     │                                               │
  │     ▼                                               │
  │  THOUGHT: "I have enough context to answer"         │
  │     │                                               │
  │     ▼                                               │
  │  FINAL ANSWER: full response to user                │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  Trace saved to LangSmith (auto via env) + backend BridgeDB (explicit)

TOOLS:
  1. search_knowledge_base  — FAISS semantic search over 5 support docs
  2. get_ticket_status       — Mock ticket DB lookup
  3. create_support_ticket   — Creates a new ticket record

MEMORY:
  ConversationBufferWindowMemory(k=5) — keeps last 5 turns in context.

TRACING:
  LangSmith: LANGCHAIN_TRACING_V2=true → all runs auto-traced to cloud
  BridgeDB:  TracedAgent wraps invoke() to save each call to SQLite
"""

import os, sys, time, uuid
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# File lives at: phase3_poc/agents/langchain_agent/agent.py
# parents[0] = langchain_agent/
# parents[1] = agents/
# parents[2] = phase3_poc/
# parents[3] = lyzr_comparison/   ← ROOT
ROOT   = Path(__file__).resolve().parents[3]
VS_DIR = Path(__file__).parent / "knowledge_base" / "vectorstore"

sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── LangChain core ────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain import hub

# ── Backend DB (optional — only needed when running inside the dashboard) ─────
try:
    from phase3_poc.backend.db import BridgeDB
except ImportError:
    BridgeDB = None

MODEL = "gpt-4o-mini"

# ══════════════════════════════════════════════════════════════════════════════
# TICKET DATABASE (mock)
# ══════════════════════════════════════════════════════════════════════════════
TICKET_DB = {
    "TKT-001": {"status": "resolved",  "issue": "Password reset not working",         "priority": "high",   "user": "alice@acme.io"},
    "TKT-002": {"status": "open",      "issue": "Cannot export analytics report",     "priority": "medium", "user": "bob@acme.io"},
    "TKT-003": {"status": "pending",   "issue": "SSO configuration help needed",      "priority": "high",   "user": "charlie@acme.io"},
    "TKT-004": {"status": "resolved",  "issue": "Billing invoice discrepancy",        "priority": "low",    "user": "dana@acme.io"},
    "TKT-005": {"status": "escalated", "issue": "API rate limit exceeded repeatedly", "priority": "high",   "user": "eve@acme.io"},
    "TKT-006": {"status": "open",      "issue": "Salesforce sync stopped",            "priority": "medium", "user": "frank@acme.io"},
    "TKT-007": {"status": "pending",   "issue": "MFA backup codes not working",       "priority": "medium", "user": "grace@acme.io"},
}

_ticket_counter = [8]

# ══════════════════════════════════════════════════════════════════════════════
# VECTORSTORE LOADER
# ══════════════════════════════════════════════════════════════════════════════
_vs_cache: FAISS | None = None

def _load_vectorstore() -> FAISS:
    global _vs_cache
    if _vs_cache is not None:
        return _vs_cache
    if not (VS_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"Vector store not found at {VS_DIR}. "
            "Run knowledge_base/build_vectorstore.py first."
        )
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    _vs_cache = FAISS.load_local(
        str(VS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return _vs_cache


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the ACME support knowledge base for product, pricing, troubleshooting,
    policy, and FAQ information.

    Use this tool for any product question. Embeds the query into a 1536-d vector
    and finds the 4 most similar document chunks via FAISS L2 search.

    Args:
        query: Natural-language question or keyword phrase.
    """
    vs = _load_vectorstore()
    results = vs.similarity_search_with_score(query, k=4)

    if not results:
        return "No relevant documentation found for this query."

    lines = [f"Found {len(results)} relevant documentation sections:\n"]
    for i, (doc, score) in enumerate(results, 1):
        source  = Path(doc.metadata.get("source", "unknown")).name.replace(".txt", "")
        quality = "strong" if score < 0.9 else "moderate" if score < 1.2 else "weak"
        lines.append(f"[{i}] Source: {source} (relevance: {quality})")
        lines.append(doc.page_content.strip())
        lines.append("")

    return "\n".join(lines)


@tool
def get_ticket_status(ticket_id: str) -> str:
    """
    Look up the status and details of a support ticket by ticket ID (e.g. TKT-003).

    Args:
        ticket_id: Ticket identifier in format TKT-NNN.
    """
    tid = ticket_id.strip().upper()
    if tid not in TICKET_DB:
        return (f"Ticket {tid} not found. Valid IDs: "
                + ", ".join(sorted(TICKET_DB.keys())))
    t = TICKET_DB[tid]
    return (
        f"Ticket {tid}\n"
        f"  Status:   {t['status'].upper()}\n"
        f"  Issue:    {t['issue']}\n"
        f"  Priority: {t['priority'].upper()}\n"
        f"  User:     {t['user']}"
    )


@tool
def create_support_ticket(issue_description: str, priority: str = "medium") -> str:
    """
    Create a new support ticket for issues that cannot be resolved via documentation.

    Use when user describes a new problem or explicitly asks to open a ticket.

    Args:
        issue_description: Clear description of the customer's problem.
        priority: 'low', 'medium', or 'high'. Defaults to 'medium'.
    """
    priority = priority.lower() if priority.lower() in {"low", "medium", "high"} else "medium"

    ticket_id = f"TKT-{_ticket_counter[0]:03d}"
    _ticket_counter[0] += 1

    TICKET_DB[ticket_id] = {
        "status":   "open",
        "issue":    issue_description,
        "priority": priority,
        "user":     "chat_user",
    }
    sla = ("1 hour (P1 SLA)" if priority == "high"
           else "8 hours (P2 SLA)" if priority == "medium"
           else "3 business days (P4 SLA)")
    return (
        f"Ticket created successfully!\n"
        f"  ID:        {ticket_id}\n"
        f"  Issue:     {issue_description}\n"
        f"  Priority:  {priority.upper()}\n"
        f"  Status:    OPEN\n"
        f"  Next steps: Our support team will contact you within {sla}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT (v1 — base version)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful, professional customer support agent for ACME SaaS Platform.

Your capabilities:
1. Search the knowledge base to answer questions about ACME products, pricing, features,
   troubleshooting, and policies.
2. Look up support ticket status by ticket ID.
3. Create new support tickets for issues that need escalation.

Guidelines:
- Always search the knowledge base FIRST when answering product questions.
- Be concise but complete. Include specific steps when giving instructions.
- If the knowledge base doesn't have the answer, acknowledge that and offer to create a ticket.
- Be empathetic — customers may be frustrated. Acknowledge their issue before diving into solutions.
- Never make up information. If uncertain, say so and suggest creating a ticket.
- Format multi-step instructions as numbered lists for clarity.

When a user mentions a ticket ID (e.g., TKT-003), always look it up first.
When a user describes a new problem that cannot be resolved, offer to create a ticket.
"""


# ══════════════════════════════════════════════════════════════════════════════
# AGENT FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(db=None, agent_version: str = "v1",
                system_prompt: str = None) -> "TracedAgent":
    """
    Build a TracedAgent wrapping LangChain's ReAct AgentExecutor.

    system_prompt: override to inject a different prompt version (for prompt A/B testing).
    """
    llm   = ChatOpenAI(model=MODEL, temperature=0,
                       openai_api_key=os.getenv("OPENAI_API_KEY"))
    tools = [search_knowledge_base, get_ticket_status, create_support_ticket]

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        output_key="output",
        return_messages=False,
    )

    try:
        base_prompt = hub.pull("hwchase17/react")
    except Exception:
        from langchain_core.prompts import PromptTemplate
        base_prompt = PromptTemplate.from_template(
            "Answer the following questions as best you can. "
            "You have access to the following tools:\n\n{tools}\n\n"
            "Use the following format:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Begin!\n\nChat History:\n{chat_history}\n\n"
            "Question: {input}\nThought:{agent_scratchpad}"
        )

    agent    = create_react_agent(llm, tools, base_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
        return_intermediate_steps=True,
    )

    return TracedAgent(
        executor,
        db=db,
        agent_version=agent_version,
        system_prompt=system_prompt or SYSTEM_PROMPT,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRACED AGENT WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class TracedAgent:
    """
    Wraps AgentExecutor and auto-saves each call as a trace to BridgeDB.
    Interceptor pattern: agent code is untouched.
    """

    def __init__(self, executor: AgentExecutor, db=None,
                 agent_version: str = "v1", system_prompt: str = ""):
        self.executor      = executor
        self.db            = db
        self.agent_version = agent_version
        self.system_prompt = system_prompt

    def chat(self, message: str, user_id: str = "anon",
             session_id: str | None = None) -> dict:
        """
        Run the agent and return:
          { response, trace_id, latency_ms, steps, status }
        """
        trace_id   = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        start_ms   = time.time() * 1000

        try:
            result = self.executor.invoke({"input": message})
            output = result.get("output", "")
            steps  = result.get("intermediate_steps", [])
            status = "success"
        except Exception as e:
            output = f"Sorry, I encountered an error: {str(e)}"
            steps  = []
            status = "error"

        latency_ms  = round(time.time() * 1000 - start_ms, 1)
        token_count = (len(message) + len(output)) // 4

        if self.db is not None:
            self.db.save_trace({
                "id":            trace_id,
                "project":       "lyzr-evalops-live",
                "platform":      "langchain",
                "agent_version": self.agent_version,
                "input":         message,
                "output":        output,
                "latency_ms":    latency_ms,
                "token_count":   token_count,
                "status":        status,
                "user_id":       user_id,
                "session_id":    session_id,
                "timestamp":     datetime.utcnow().isoformat(),
            })

        steps_summary = [
            {
                "tool":        action.tool,
                "tool_input":  str(action.tool_input),
                "observation": str(observation)[:500],
            }
            for action, observation in steps
        ]

        return {
            "response":   output,
            "trace_id":   trace_id,
            "latency_ms": latency_ms,
            "steps":      steps_summary,
            "status":     status,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ══════════════════════════════════════════════════════════════════════════════
_agent_instance: TracedAgent | None = None

def get_agent(db=None, agent_version: str = "v1",
              system_prompt: str = None) -> TracedAgent:
    """Return (and cache) a single TracedAgent instance."""
    global _agent_instance
    if _agent_instance is None:
        print("[LangChain Agent] Initializing agent and loading vector store...")
        _agent_instance = build_agent(db=db, agent_version=agent_version,
                                      system_prompt=system_prompt)
        print("[LangChain Agent] Ready.")
    return _agent_instance


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  LangChain Agent — Standalone Test")
    print("=" * 65)

    agent = build_agent(db=None, agent_version="v1-test")

    for q in [
        "How do I configure SSO with Okta?",
        "What's the price of the Growth plan?",
        "I'm getting ERR-SAML-001, how do I fix it?",
        "What's the status of ticket TKT-003?",
        "Can you open a ticket for my API rate limit issue?",
    ]:
        print(f"\n{'─' * 60}\nUSER: {q}\n{'─' * 60}")
        result = agent.chat(q, user_id="test_user")
        print(f"\nAGENT: {result['response']}")
        print(f"[{result['latency_ms']}ms | tools: {[s['tool'] for s in result['steps']]}]")
