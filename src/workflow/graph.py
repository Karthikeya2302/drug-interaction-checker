from langgraph.graph import StateGraph, END
from src.workflow.state import DrugInteractionState
from src.workflow.nodes import (
    extract_drugs,
    faiss_search,
    mcp_search,
    assess_severity,
    generate_response
)

def build_graph():
    """
    Assembles all 5 nodes into a LangGraph workflow.
    Nodes 2 and 3 run in parallel.
    """

    # Initialize the graph with our state definition
    graph = StateGraph(DrugInteractionState)

    # ── Add all 5 nodes ───────────────────────────────────
    graph.add_node("extract_drugs", extract_drugs)
    graph.add_node("faiss_search", faiss_search)
    graph.add_node("mcp_search", mcp_search)
    graph.add_node("assess_severity", assess_severity)
    graph.add_node("generate_response", generate_response)

    # ── Define the flow ───────────────────────────────────
    # Start at Node 1
    graph.set_entry_point("extract_drugs")

    # Node 1 → Node 2 AND Node 3 in parallel
    graph.add_edge("extract_drugs", "faiss_search")
    graph.add_edge("extract_drugs", "mcp_search")

    # Both Node 2 and Node 3 → Node 4
    graph.add_edge("faiss_search", "assess_severity")
    graph.add_edge("mcp_search", "assess_severity")

    # Node 4 → Node 5
    graph.add_edge("assess_severity", "generate_response")

    # Node 5 → End
    graph.add_edge("generate_response", END)

    # Compile the graph into a runnable object
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