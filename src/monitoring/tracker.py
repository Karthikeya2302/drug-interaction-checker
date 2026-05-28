from langsmith.run_trees import RunTree
from config import config


def log_interaction(
    query: str,
    drug_1: str,
    drug_2: str,
    faiss_hits: int,
    pubmed_hits: int,
    fda_found: bool,
    severity: str,
    confidence: float,
    response_time: float,
):
    """Log a drug interaction query to LangSmith as a RunTree span."""
    try:
        run = RunTree(
            name="drug-interaction-check",
            run_type="chain",
            project_name=config.LANGSMITH_PROJECT,
            inputs={
                "query": query,
                "drug_1": drug_1,
                "drug_2": drug_2,
            },
        )
        run.end(outputs={
            "severity": severity,
            "confidence": confidence,
            "faiss_hits": faiss_hits,
            "pubmed_hits": pubmed_hits,
            "fda_found": fda_found,
            "response_time_seconds": response_time,
        })
        run.post()
    except Exception as e:
        print(f"  LangSmith logging failed: {e}")


def log_interaction_check(
    user_query: str,
    drug_1: str,
    drug_2: str,
    severity: str,
    confidence: float,
    faiss_score: float,
    pubmed_count: int,
    web_count: int,
    response_time: float,
    validated: bool,
):
    """Adapter called by graph.py — maps its kwargs into log_interaction()."""
    log_interaction(
        query=user_query,
        drug_1=drug_1,
        drug_2=drug_2,
        faiss_hits=1 if faiss_score > 0 else 0,
        pubmed_hits=pubmed_count,
        fda_found=web_count > 0,
        severity=severity,
        confidence=confidence,
        response_time=response_time,
    )


def get_average_metrics() -> dict:
    """Metrics now live in LangSmith — returns empty so the footer shows '—'."""
    return {}
