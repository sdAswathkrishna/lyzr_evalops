"""LangChain ReAct agent with LangSmith tracing — the baseline."""

import os
import time
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain import hub
from langsmith import Client, traceable

load_dotenv()

from .tools import (
    KNOWLEDGE_BASE, TICKET_DB, TEST_QUERIES,
    search_knowledge_base as _skb,
    get_ticket_status     as _gts,
    create_ticket         as _ct,
)

@tool
def search_knowledge_base(query: str) -> str:
    """Search product knowledge base. Input: keywords like 'billing', 'sso', 'api limit'."""
    return _skb(query)

@tool
def get_ticket_status(ticket_id: str) -> str:
    """Look up a support ticket status by ID, e.g. TKT-003."""
    return _gts(ticket_id)

@tool
def create_ticket(issue: str, severity: str = "medium") -> str:
    """Create a support ticket. severity: low/medium/high."""
    return _ct(issue, severity)


def build_langsmith_agent() -> AgentExecutor:
    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools  = [search_knowledge_base, get_ticket_status, create_ticket]
    prompt = hub.pull("hwchase17/react")
    agent  = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False,
                         handle_parsing_errors=True, max_iterations=6)


@traceable(name="ls-support-session", tags=["poc", "v1"],
           metadata={"platform": "langsmith", "agent_version": "v1"})
def run_langsmith_query(executor: AgentExecutor, query: str, user_id: str) -> dict:
    start = time.time()
    result = executor.invoke(
        {"input": query},
        config={"metadata": {"user_id": user_id}, "tags": ["poc"]},
    )
    return {
        "input":      query,
        "output":     result["output"],
        "latency_ms": round((time.time() - start) * 1000),
        "user_id":    user_id,
    }


def run_all(queries=None) -> list[dict]:
    """Run all test queries and return results. Also logs to LangSmith."""
    executor = build_langsmith_agent()
    queries  = queries or TEST_QUERIES
    results  = []
    for query, user_id in queries:
        r = run_langsmith_query(executor, query, user_id)
        results.append(r)
        time.sleep(0.3)
    return results
