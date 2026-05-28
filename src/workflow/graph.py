from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("LANGCHAIN_API_KEY")
if api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = "drug-interaction-checker"

from langgraph.graph import StateGraph, END
from src.workflow.state import DrugInteractionState
from src.workflow.nodes import (
    extract_drugs,
    faiss_search,
    grade_retrieval,
    mcp_search,
    assess_severity,
    generate_response
)

def build_graph():
    """
    Assembles the LangGraph pipeline as a linear chain.
    extract_drugs → faiss_search → mcp_search → grade_retrieval
    → assess_severity → generate_response
    """

    graph = StateGraph(DrugInteractionState)

    # ── Register nodes ────────────────────────────────────
    graph.add_node("extract_drugs", extract_drugs)
    graph.add_node("faiss_search", faiss_search)
    graph.add_node("mcp_search", mcp_search)
    graph.add_node("grade_retrieval", grade_retrieval)
    graph.add_node("assess_severity", assess_severity)
    graph.add_node("generate_response", generate_response)

    # ── Linear pipeline — exactly these edges, nothing else ──
    graph.set_entry_point("extract_drugs")
    graph.add_edge("extract_drugs", "faiss_search")
    graph.add_edge("faiss_search", "mcp_search")
    graph.add_edge("mcp_search", "grade_retrieval")
    graph.add_edge("grade_retrieval", "assess_severity")
    graph.add_edge("assess_severity", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


# Module level app — imported by Streamlit directly
app = build_graph()


def run_interaction_check(user_query: str) -> dict:
    """
    Main entry point for checking drug interactions.
    Called by Streamlit frontend.
    """
    from src.monitoring.tracker import log_interaction_check

    initial_state = {
        "user_query": user_query,
        "drug_1_user": "",
        "drug_1_clinical": "",
        "drug_2_user": "",
        "drug_2_clinical": "",
        "drugs_validated": False,
        "validation_error": None,
        "faiss_results": [],
        "graded_faiss_results": [],
        "retrieval_quality": "unknown",
        "retrieval_corrected": False,
        "pubmed_results": [],
        "web_results": [],
        "severity": "unknown",
        "confidence": 0.0,
        "final_response": "",
        "response_time": 0.0
    }

    result = app.invoke(initial_state)

    # Log to MLflow after every query
    faiss_score = result["faiss_results"][0]["score"] if result["faiss_results"] else 0.0

    log_interaction_check(
        user_query=user_query,
        drug_1=result.get("drug_1_user", ""),
        drug_2=result.get("drug_2_user", ""),
        severity=result.get("severity", "unknown"),
        confidence=result.get("confidence", 0.0),
        faiss_score=faiss_score,
        pubmed_count=len(result.get("pubmed_results", [])),
        web_count=len(result.get("web_results", [])),
        response_time=result.get("response_time", 0.0),
        validated=result.get("drugs_validated", False)
    )

    return result

if __name__ == "__main__":
    # Quick test
    result = run_interaction_check("Can I take Aspirin with Warfarin?")
    print(result["final_response"])