"""
Dataset management for the Lyzr EvalOps bridge.
Mirrors the LangSmith dataset API surface.
"""

from .db import BridgeDB

REFERENCE_ANSWERS = {
    "How do I update my billing information?": (
        "To update your billing information, go to Settings → Billing. "
        "To change your credit card, navigate to Settings → Payment Methods. "
        "Invoices are generated on the 1st of each month."
    ),
    "What is the API rate limit for the Pro plan?": (
        "The Pro plan allows 10,000 API calls per day. "
        "If you exceed this limit you will receive an HTTP 429 error. "
        "You can request a temporary increase via Settings → API → Request Increase."
    ),
    "I need to set up SSO with Okta for my enterprise account.": (
        "SSO is available on Enterprise plans. Okta is a supported provider. "
        "Follow the configuration guide at docs.example.com/sso."
    ),
    "What's the status of ticket TKT-003?": (
        "Ticket TKT-003 is currently Pending. "
        "The issue is SSO configuration help, opened 1 day ago."
    ),
    "My app keeps getting HTTP 429 errors, can you help?": (
        "HTTP 429 means you have exceeded your API rate limit. "
        "The Pro plan limit is 10,000 calls per day. "
        "Request a temporary increase via Settings → API → Request Increase."
    ),
    "I haven't received my password reset email.": (
        "Password reset emails arrive within 2 minutes. "
        "Please check your spam folder. "
        "If the issue persists, a support ticket should be created."
    ),
    "How do I export my data to JSON format?": (
        "Go to Settings → Data → Export. "
        "JSON, CSV, and Parquet formats are supported. "
        "Exports are processed asynchronously and emailed to you."
    ),
}


class DatasetStore:
    def __init__(self, db: BridgeDB = None):
        self.db = db or BridgeDB()

    def create_from_reference(self, name: str = "tech-support-v1") -> str:
        """
        Create the evaluation dataset from curated reference answers.
        Equivalent to LangSmith's client.create_dataset() + client.create_example() loop.
        """
        dataset_id = self.db.create_dataset(
            name=name,
            description="Tech Support Agent eval dataset — curated Q&A pairs"
        )
        for question, answer in REFERENCE_ANSWERS.items():
            self.db.add_example(
                dataset_id=dataset_id,
                inputs={"input": question},
                outputs={"output": answer},
                metadata={"source": "manual_curation", "category": _categorise(question)},
            )
        return dataset_id

    def enrich_from_traces(self, dataset_name: str, traces: list[dict]) -> int:
        """
        Add production traces to the dataset as new examples.
        Equivalent to LangSmith's 'Add to Dataset' trace action.
        """
        dataset = self.db.get_dataset_by_name(dataset_name)
        if not dataset:
            return 0

        existing = {
            ex["inputs"].get("input", "")
            for ex in self.db.list_examples(dataset["id"])
        }
        added = 0
        for trace in traces:
            if trace["input"] not in existing and trace["status"] == "success":
                self.db.add_example(
                    dataset_id=dataset["id"],
                    inputs={"input": trace["input"]},
                    outputs={"output": trace["output"]},
                    metadata={
                        "source": "production_trace",
                        "source_trace_id": trace["id"],
                        "needs_review": True,
                    },
                    source_trace_id=trace["id"],
                )
                added += 1
        return added

    def list_examples(self, dataset_name: str) -> list[dict]:
        dataset = self.db.get_dataset_by_name(dataset_name)
        return self.db.list_examples(dataset["id"]) if dataset else []


def _categorise(q: str) -> str:
    q = q.lower()
    if "billing" in q or "payment" in q:       return "billing"
    if "api" in q or "429" in q or "rate" in q: return "api"
    if "sso" in q or "okta" in q:              return "sso"
    if "ticket" in q or "tkt" in q:            return "ticketing"
    if "password" in q or "reset" in q:        return "auth"
    if "export" in q or "data" in q:           return "data"
    return "general"
