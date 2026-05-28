from typing import TypedDict, Optional

class DrugInteractionState(TypedDict):
    # ── Input ────────────────────────────────────────────
    # Original query from the user
    user_query: str

    # ── Node 1 outputs ───────────────────────────────────
    # Drug names exactly as user typed them
    drug_1_user: str
    drug_2_user: str

    # Normalized clinical names for searching
    # e.g. "Aspirin" → "Acetylsalicylic acid"
    drug_1_clinical: str
    drug_2_clinical: str

    # Whether both drugs were validated as real drugs
    drugs_validated: bool

    # If validation fails, this message goes back to user
    validation_error: Optional[str]

    # ── Node 2 outputs ───────────────────────────────────
    # Top 5 results from FAISS local database
    faiss_results: list

    # ── Node 2.5 outputs (CRAG grader) ───────────────────
    # FAISS results that passed relevance grading (score >= 0.5)
    graded_faiss_results: list
    # "high" / "medium" / "low"
    retrieval_quality: str
    # True when grading discarded everything and fallback was triggered
    retrieval_corrected: bool

    # ── Node 3 outputs ───────────────────────────────────
    # Latest research papers from PubMed via MCP
    pubmed_results: list

    # FDA warnings and web results via MCP
    web_results: list

    # ── Node 4 outputs ───────────────────────────────────
    # Severity classification: major/moderate/minor/none
    severity: str

    # Confidence score 0-1
    confidence: float

    # ── Node 5 outputs ───────────────────────────────────
    # Final formatted response shown to user
    final_response: str

    response_time: float