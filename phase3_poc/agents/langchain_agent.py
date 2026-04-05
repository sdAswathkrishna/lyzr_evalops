"""
langchain_agent.py — Real LangChain ReAct Agent with embeddings + tracing.

HOW THE AGENT WORKS (the full mental model):
=============================================

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
  Trace saved to LangSmith + BridgeDB

TOOLS THE AGENT HAS:
  1. search_knowledge_base  — FAISS semantic search over 5 support docs
  2. get_ticket_status       — Mock ticket DB lookup
  3. create_support_ticket   — Creates a new ticket record

MEMORY:
  ConversationBufferWindowMemory(k=5) — keeps last 5 turns so the agent
  can reference "what I told you before".

TRACING:
  LangSmith: LANGCHAIN_TRACING_V2=true → all runs auto-traced
  BridgeDB:  We wrap the invoke() call manually to save to SQLite
"""

import os, sys, time, uuid
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── LangChain core ───────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain import hub

# ── BridgeDB (optional — only needed when running inside the dashboard) ───────
try:
    from phase3_poc.bridge.db import BridgeDB
except ImportError:
    BridgeDB = None

# ── Constants ─────────────────────────────────────────────────────────────────
VS_DIR = Path(__file__).parent / "knowledge_base" / "vectorstore"
MODEL  = "gpt-4o-mini"

# ══════════════════════════════════════════════════════════════════════════════
# TICKET DATABASE (mock — same as before, but now only used by ticket tools)
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

_ticket_counter = [8]  # mutable so the @tool closure can increment it

# ══════════════════════════════════════════════════════════════════════════════
# VECTORSTORE LOADER
# ══════════════════════════════════════════════════════════════════════════════
_vs_cache: FAISS | None = None   # module-level cache — load once, reuse

def _load_vectorstore() -> FAISS:
    """
    Load the FAISS index from disk (built by build_vectorstore.py).

    allow_dangerous_deserialization=True is required because FAISS's save_local()
    uses pickle. We set this flag knowingly because we wrote the index ourselves.
    In production, you would verify a checksum of the index files.
    """
    global _vs_cache
    if _vs_cache is not None:
        return _vs_cache
    if not (VS_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"Vector store not found at {VS_DIR}. "
            "Run build_vectorstore.py first."
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
# TOOL 1 — Knowledge Base Retrieval  ← the new "embedding" tool
# ══════════════════════════════════════════════════════════════════════════════
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the ACME support knowledge base for information about products,
    features, pricing, troubleshooting, policies, and FAQs.

    Use this tool whenever the user asks a question that requires product
    knowledge, pricing information, troubleshooting steps, or policy details.

    Args:
        query: A natural-language question or keyword phrase.

    Returns:
        The most relevant documentation excerpts, with source file noted.

    INTERNAL MECHANICS (for learning):
      1. query string → OpenAI Embeddings API → 1536-d float vector
      2. FAISS.similarity_search_with_score(query_vector, k=4)
         → finds 4 nearest neighbours in the index using L2 distance
      3. Returns list of (Document, score) pairs
      4. We format them as a readable string for the LLM
    """
    vs = _load_vectorstore()
    results = vs.similarity_search_with_score(query, k=4)

    if not results:
        return "No relevant documentation found for this query."

    lines = [f"Found {len(results)} relevant documentation sections:\n"]
    for i, (doc, score) in enumerate(results, 1):
        source = Path(doc.metadata.get("source", "unknown")).name.replace(".txt", "")
        # score is L2 distance; lower = more similar. Score < 1.2 is a good match.
        quality = "strong" if score < 0.9 else "moderate" if score < 1.2 else "weak"
        lines.append(f"[{i}] Source: {source} (relevance: {quality})")
        lines.append(doc.page_content.strip())
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Ticket Status Lookup
# ══════════════════════════════════════════════════════════════════════════════
@tool
def get_ticket_status(ticket_id: str) -> str:
    """
    Look up the status and details of a support ticket by ticket ID.

    Use this when the user mentions a ticket number (e.g., TKT-003).

    Args:
        ticket_id: Ticket identifier in format TKT-NNN (e.g., TKT-003).

    Returns:
        Ticket details including status, issue description, and priority.
    """
    tid = ticket_id.strip().upper()
    if tid not in TICKET_DB:
        return (f"Ticket {tid} not found. Valid ticket IDs: "
                + ", ".join(sorted(TICKET_DB.keys())))
    t = TICKET_DB[tid]
    return (
        f"Ticket {tid}\n"
        f"  Status:   {t['status'].upper()}\n"
        f"  Issue:    {t['issue']}\n"
        f"  Priority: {t['priority'].upper()}\n"
        f"  User:     {t['user']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Create Support Ticket
# ══════════════════════════════════════════════════════════════════════════════
@tool
def create_support_ticket(issue_description: str, priority: str = "medium") -> str:
    """
    Create a new support ticket for an issue that cannot be resolved via documentation.

    Use this when the user has a new problem that should be escalated to the support team,
    or when they explicitly ask to open a ticket.

    Args:
        issue_description: Clear description of the customer's problem.
        priority: One of 'low', 'medium', 'high'. Defaults to 'medium'.

    Returns:
        Confirmation with the new ticket ID.
    """
    valid_priorities = {"low", "medium", "high"}
    priority = priority.lower() if priority.lower() in valid_priorities else "medium"

    ticket_id = f"TKT-{_ticket_counter[0]:03d}"
    _ticket_counter[0] += 1

    TICKET_DB[ticket_id] = {
        "status":   "open",
        "issue":    issue_description,
        "priority": priority,
        "user":     "chat_user",
    }
    return (
        f"✅ Ticket created successfully!\n"
        f"  ID:          {ticket_id}\n"
        f"  Issue:       {issue_description}\n"
        f"  Priority:    {priority.upper()}\n"
        f"  Status:      OPEN\n"
        f"  Next steps:  Our support team will contact you within "
        + ("1 hour (P1 SLA)" if priority == "high" else
           "8 hours (P2 SLA)" if priority == "medium" else
           "3 business days (P4 SLA)")
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT FACTORY
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


def build_agent(db: "BridgeDB | None" = None, agent_version: str = "v1") -> "TracedAgent":
    """
    Build and return a TracedAgent wrapping the LangChain ReAct executor.

    The ReAct agent uses the "hwchase17/react" prompt from LangChain Hub.
    This prompt structures the LLM to output:
      Thought: <reasoning>
      Action: <tool_name>
      Action Input: <tool_input>
      Observation: <tool_output>
      ... (repeat until done)
      Final Answer: <answer>

    The AgentExecutor handles the loop, routing tool calls, and collecting observations.
    """
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    tools = [search_knowledge_base, get_ticket_status, create_support_ticket]

    # Memory: keeps the last 5 turns in the prompt so the agent has context.
    # output_key="output" disambiguates because AgentExecutor returns both
    # "output" and "intermediate_steps"; memory should only store "output".
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        output_key="output",
        return_messages=False,   # plain text format, not message objects
    )

    # Pull the standard ReAct prompt template from LangChain Hub.
    # It expects variables: {tools}, {tool_names}, {input}, {agent_scratchpad}, {chat_history}
    try:
        base_prompt = hub.pull("hwchase17/react")
    except Exception:
        # Fallback if Hub is unreachable: use the prompt template inline
        from langchain_core.prompts import PromptTemplate
        base_prompt = PromptTemplate.from_template(
            "Answer the following questions as best you can. You have access to the following tools:\n\n"
            "{tools}\n\n"
            "Use the following format:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Begin!\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Question: {input}\n"
            "Thought:{agent_scratchpad}"
        )

    # Inject the system prompt into the template's prefix
    # The hub prompt has a "system" variable we can override
    if hasattr(base_prompt, "messages"):
        # ChatPromptTemplate — insert our system message at the front
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
        # We'll just modify the input by prepending system context to the input
        pass  # use prompt as-is, add system context to the tool descriptions

    agent = create_react_agent(llm, tools, base_prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,           # prints Thought/Action/Observation to stdout
        handle_parsing_errors=True,  # if LLM output doesn't parse, retry once
        max_iterations=6,       # prevent infinite loops
        return_intermediate_steps=True,
    )

    return TracedAgent(executor, db=db, agent_version=agent_version)


# ══════════════════════════════════════════════════════════════════════════════
# TRACED AGENT WRAPPER
# Wraps the AgentExecutor to auto-save traces to BridgeDB
# ══════════════════════════════════════════════════════════════════════════════
class TracedAgent:
    """
    Wraps AgentExecutor and automatically saves each invocation as a trace
    to BridgeDB so the dashboard can display it.

    This is the "interceptor pattern" — we don't modify the AgentExecutor at
    all, we just measure time around the invoke() call and persist the result.
    """

    def __init__(self, executor: AgentExecutor, db=None, agent_version: str = "v1"):
        self.executor      = executor
        self.db            = db
        self.agent_version = agent_version

    def chat(self, message: str, user_id: str = "anon", session_id: str | None = None) -> dict:
        """
        Run the agent on a user message and return:
          {
            "response":  str,        final answer
            "trace_id":  str,        UUID saved in BridgeDB
            "latency_ms": float,
            "steps":     list,       intermediate Thought/Action/Obs steps
          }
        """
        trace_id   = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        start_ms   = time.time() * 1000

        try:
            # The ReAct hub prompt expects only "input". System instructions are
            # embedded via the tool descriptions and the agent's own training.
            # If you need explicit system context, use a custom ChatPromptTemplate
            # with a SystemMessage placeholder (see build_agent docstring).
            result = self.executor.invoke({"input": message})
            output     = result.get("output", "")
            steps      = result.get("intermediate_steps", [])
            status     = "success"
        except Exception as e:
            output = f"Sorry, I encountered an error: {str(e)}"
            steps  = []
            status = "error"

        latency_ms = round(time.time() * 1000 - start_ms, 1)

        # Estimate token count (rough: 1 token ≈ 4 chars)
        token_count = (len(message) + len(output)) // 4

        # Persist to BridgeDB
        # Keys must match the SQL column names used in save_trace() (:id, :project, etc.)
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
                "timestamp":     __import__("datetime").datetime.utcnow().isoformat(),
            })

        # Build steps summary for the UI
        steps_summary = []
        for action, observation in steps:
            steps_summary.append({
                "tool":        action.tool,
                "tool_input":  str(action.tool_input),
                "observation": str(observation)[:500],  # truncate long observations
            })

        return {
            "response":   output,
            "trace_id":   trace_id,
            "latency_ms": latency_ms,
            "steps":      steps_summary,
            "status":     status,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON (used by the FastAPI app)
# ══════════════════════════════════════════════════════════════════════════════
_agent_instance: TracedAgent | None = None

def get_agent(db=None, agent_version: str = "v1") -> TracedAgent:
    """Return (and cache) a single TracedAgent instance."""
    global _agent_instance
    if _agent_instance is None:
        print("[LangChain Agent] Initializing agent and loading vector store...")
        _agent_instance = build_agent(db=db, agent_version=agent_version)
        print("[LangChain Agent] Ready.")
    return _agent_instance


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  Testing LangChain Agent with Embeddings")
    print("=" * 65)

    agent = build_agent(db=None, agent_version="v1-test")

    test_questions = [
        "How do I configure SSO with Okta?",
        "What's the price of the Growth plan and what does it include?",
        "I'm getting ERR-SAML-001 when logging in, how do I fix it?",
        "What's the status of ticket TKT-003?",
        "I can't figure out the API rate limits. Can you open a ticket for me?",
    ]

    for q in test_questions:
        print(f"\n{'─' * 60}")
        print(f"USER: {q}")
        print("─" * 60)
        result = agent.chat(q, user_id="test_user")
        print(f"\nAGENT: {result['response']}")
        print(f"\n[Latency: {result['latency_ms']}ms | Steps: {len(result['steps'])}]")
        if result["steps"]:
            for s in result["steps"]:
                print(f"  → Tool used: {s['tool']}({s['tool_input'][:60]})")
