"""Shared mock data and tool functions — used by both LangSmith and Lyzr agents."""

KNOWLEDGE_BASE = {
    "billing":   "Billing is managed via Settings → Billing. Invoices on the 1st of each month. Credit card: Settings → Payment Methods.",
    "api_limit": "Free: 100/day. Pro: 10,000/day. Enterprise: unlimited. 429 = rate limit. Increase via Settings → API → Request Increase.",
    "reset":     "Click 'Forgot Password' on login page. Email arrives in 2 min. Check spam if not received.",
    "sso":       "SSO on Enterprise plans only. Providers: Okta, Azure AD, Google Workspace. Guide: docs.example.com/sso",
    "export":    "Settings → Data → Export. Formats: CSV, JSON, Parquet. Async — results emailed.",
}
TICKET_DB = {
    "TKT-001": {"status": "Open",     "issue": "API returning 500 errors",       "age": "2 hours"},
    "TKT-002": {"status": "Resolved", "issue": "Billing overcharge",             "age": "3 days"},
    "TKT-003": {"status": "Pending",  "issue": "SSO configuration help",         "age": "1 day"},
    "TKT-004": {"status": "Open",     "issue": "Password reset not received",    "age": "30 mins"},
}
TEST_QUERIES = [
    ("How do I update my billing information?",             "user_001"),
    ("What is the API rate limit for the Pro plan?",        "user_002"),
    ("I need to set up SSO with Okta for my enterprise account.", "user_003"),
    ("What's the status of ticket TKT-003?",               "user_004"),
    ("My app keeps getting HTTP 429 errors, can you help?", "user_005"),
    ("I haven't received my password reset email.",         "user_006"),
    ("How do I export my data to JSON format?",             "user_007"),
]

def search_knowledge_base(query: str) -> str:
    q = query.lower()
    for key, article in KNOWLEDGE_BASE.items():
        if key in q or any(w in q for w in key.split("_")):
            return f"[KB: {key}] {article}"
    return "No article found. Direct user to docs.example.com."

def get_ticket_status(ticket_id: str) -> str:
    tid = ticket_id.strip().upper()
    if tid in TICKET_DB:
        t = TICKET_DB[tid]
        return f"{tid}: {t['status']} | {t['issue']} | {t['age']} ago"
    return f"{tid} not found. Please verify the ticket number."

def create_ticket(issue: str, severity: str = "medium") -> str:
    import random
    tid = f"TKT-{random.randint(100, 999)}"
    eta = {"low": "3-5 days", "medium": "24 hours", "high": "2 hours"}.get(severity.lower(), "24h")
    return f"Created {tid}: {issue} | severity={severity} | ETA={eta}"
