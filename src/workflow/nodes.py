import time
import json
from groq import Groq
from src.rag.retriever import DrugInteractionRetriever
from src.workflow.state import DrugInteractionState
from config import config

# Initialize Groq client once — reused across all nodes
groq_client = Groq(api_key=config.GROQ_API_KEY)

# Initialize retriever once — loads FAISS index into memory once
retriever = DrugInteractionRetriever()


def call_llm(prompt: str, system: str = "") -> str:
    """Helper function to call Groq LLaMA — used by all nodes."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = groq_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=0.1,  # Low temperature = more consistent, less creative
        max_tokens=1000
    )
    return response.choices[0].message.content


# ── Node 1 ────────────────────────────────────────────────────────────────
def extract_drugs(state: DrugInteractionState) -> dict:
    """
    Extract drug names from user query.
    Normalize to clinical names.
    Validate both are real drugs.
    """
    print("Node 1: Extracting drug names...")

    system = """You are a clinical pharmacology expert.
Your job is to extract and normalize drug names from user queries.
Always respond with valid JSON only. No extra text."""

    prompt = f"""Extract the two drug names from this query and normalize them to their clinical names.

Query: {state['user_query']}

Return this exact JSON format:
{{
    "drug_1_user": "name as user typed",
    "drug_1_clinical": "official clinical name",
    "drug_2_user": "name as user typed",
    "drug_2_clinical": "official clinical name",
    "validated": true or false,
    "error": "error message if not valid drugs, else null"
}}

Examples of normalization:
- Aspirin → Acetylsalicylic acid
- Tylenol → Acetaminophen
- Advil → Ibuprofen"""

    response = call_llm(prompt, system)

    # Parse JSON response from LLaMA
    try:
        # Clean response in case LLaMA adds markdown backticks
        clean = response.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean)

        return {
            "drug_1_user": data.get("drug_1_user", ""),
            "drug_1_clinical": data.get("drug_1_clinical", ""),
            "drug_2_user": data.get("drug_2_user", ""),
            "drug_2_clinical": data.get("drug_2_clinical", ""),
            "drugs_validated": data.get("validated", False),
            "validation_error": data.get("error", None)
        }
    except json.JSONDecodeError:
        return {
            "drugs_validated": False,
            "validation_error": "Could not parse drug names. Please try again."
        }


# ── Node 2 ────────────────────────────────────────────────────────────────
def faiss_search(state: DrugInteractionState) -> dict:
    """
    Search local FAISS index using clinical drug names.
    Returns top 5 most relevant interactions.
    """
    print("Node 2: Searching local database...")

    # Stop if drugs weren't validated in Node 1
    if not state.get("drugs_validated"):
        return {"faiss_results": []}

    # Build search query from clinical names
    query = f"{state['drug_1_clinical']} {state['drug_2_clinical']} interaction"

    results = retriever.retrieve(query, top_k=config.TOP_K_RETRIEVAL)

    print(f"  Found {len(results)} local results")
    return {"faiss_results": results}


# ── Node 2.5 ──────────────────────────────────────────────────────────────
def grade_retrieval(state: DrugInteractionState) -> dict:
    """CRAG grader: filter FAISS results by similarity score."""
    print("Node 2.5: Grading retrieval quality...")

    faiss_results = state.get("faiss_results", [])

    relevant = [r for r in faiss_results if r["score"] >= 0.7]
    partial  = [r for r in faiss_results if 0.5 <= r["score"] < 0.7]
    noise    = [r for r in faiss_results if r["score"] < 0.5]

    print(f"  Relevant (>=0.7): {len(relevant)}")
    print(f"  Partial (0.5-0.7): {len(partial)}")
    print(f"  Noise (<0.5): {len(noise)} — discarded")

    if len(relevant) >= 2:
        graded = relevant
        quality = "high"
        corrected = False
    elif len(relevant) + len(partial) >= 1:
        graded = relevant + partial
        quality = "medium"
        corrected = False
    else:
        graded = []
        quality = "low"
        corrected = True
        print("  ⚠️ Low quality retrieval — flagging for live source fallback")

    return {
        **state,
        "graded_faiss_results": graded,
        "retrieval_quality": quality,
        "retrieval_corrected": corrected
    }


# ── Node 3 ────────────────────────────────────────────────────────────────
def mcp_search(state: DrugInteractionState) -> dict:
    """
    Search PubMed and FDA live APIs.
    Runs in parallel with Node 2.
    """
    print("Node 3: Searching live sources...")

    if not state.get("drugs_validated"):
        return {"pubmed_results": [], "web_results": []}

    drug_1 = state["drug_1_clinical"]
    drug_2 = state["drug_2_clinical"]

    pubmed_results = search_pubmed(drug_1, drug_2)
    web_results = search_fda(drug_1, drug_2)

    print(f"  PubMed: {len(pubmed_results)} papers found")
    print(f"  FDA: {len(web_results)} results found")

    return {
        "pubmed_results": pubmed_results,
        "web_results": web_results
    }


def search_pubmed(drug_1: str, drug_2: str) -> list:
    """Search PubMed via NCBI Entrez API — free, no key required."""
    import requests

    query = f"{drug_1} {drug_2} drug interaction"
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    try:
        # Step 1 — get paper IDs
        search_response = requests.get(
            f"{base_url}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": 3,
                "retmode": "json",
                "sort": "relevance"
            },
            timeout=10
        )
        ids = search_response.json()["esearchresult"]["idlist"]

        if not ids:
            return []

        # Step 2 — get paper details
        fetch_response = requests.get(
            f"{base_url}/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json"
            },
            timeout=10
        )
        data = fetch_response.json()

        results = []
        for paper_id in ids:
            paper = data["result"].get(paper_id, {})
            results.append({
                "title": paper.get("title", ""),
                "source": "PubMed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
                "year": paper.get("pubdate", "")[:4]
            })

        return results

    except Exception as e:
        print(f"  PubMed search failed: {e}")
        return []


def _fetch_fda_label(drug_name: str):
    """
    Try three OpenFDA fields in lowercase order.
    Returns (label_name, interaction_text) or (None, "No FDA label found").
    """
    import requests

    drug_lower = drug_name.lower()
    url = "https://api.fda.gov/drug/label.json"

    attempts = [
        ("generic_name",  f'openfda.generic_name:"{drug_lower}"'),
        ("brand_name",    f'openfda.brand_name:"{drug_lower}"'),
        ("substance_name", f'openfda.substance_name:"{drug_lower}"'),
    ]

    for field, query in attempts:
        try:
            print(f"  FDA query: {query}")
            r = requests.get(url, params={"search": query, "limit": 1}, timeout=10)
            print(f"  FDA raw response ({r.status_code}): {r.text[:500]}")
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    data = results[0]
                    name = data.get("openfda", {}).get(field, ["Unknown"])[0]
                    content = data.get("drug_interactions", [""])[0][:300]
                    return name, content
        except Exception:
            pass

    return None, "No FDA label found"


def search_fda(drug_1: str, drug_2: str) -> list:
    """Search FDA drug label database — free, no key required."""
    name, content = _fetch_fda_label(drug_1)

    if name is None:
        return [{
            "title": "FDA: No label found",
            "content": "FDA: No label found",
            "source": "FDA",
            "url": "https://api.fda.gov"
        }]

    return [{
        "title": f"FDA Label: {name}",
        "content": content,
        "source": "FDA",
        "url": "https://api.fda.gov"
    }]

# ── Node 4 ────────────────────────────────────────────────────────────────
def assess_severity(state: DrugInteractionState) -> dict:
    """
    Combine FAISS + MCP results.
    Classify interaction severity using LLaMA.
    """
    print("Node 4: Assessing severity...")

    if not state.get("drugs_validated"):
        return {"severity": "unknown", "confidence": 0.0}

    # Combine all evidence — prefer graded results, fall back to raw FAISS
    graded = state.get("graded_faiss_results") or state.get("faiss_results", [])
    faiss_text = "\n".join([r["interaction"] for r in graded])
    pubmed_text = "\n".join([r["title"] for r in state.get("pubmed_results", [])])
    web_text = "\n".join([r.get("content", "") for r in state.get("web_results", [])])

    system = """You are a clinical pharmacist assessing drug interaction severity.
Always respond with valid JSON only."""

    prompt = f"""Assess the severity of interaction between {state['drug_1_user']} and {state['drug_2_user']}.

Local Database Evidence:
{faiss_text}

PubMed Research:
{pubmed_text}

Web/FDA Sources:
{web_text}

Classify severity and return this exact JSON:
{{
    "severity": "major" or "moderate" or "minor" or "none",
    "confidence": 0.0 to 1.0,
    "reasoning": "one sentence explanation"
}}"""

    response = call_llm(prompt, system)

    try:
        clean = response.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean)
        return {
            "severity": data.get("severity", "unknown"),
            "confidence": float(data.get("confidence", 0.0))
        }
    except json.JSONDecodeError:
        return {"severity": "unknown", "confidence": 0.0}


# ── Node 5 ────────────────────────────────────────────────────────────────
def generate_response(state: DrugInteractionState) -> dict:
    """Generate final structured clinical report."""
    print("Node 5: Generating final response...")

    start_time = time.time()

    if not state.get("drugs_validated"):
        return {
            "final_response": f"❌ {state.get('validation_error', 'Could not identify the drugs. Please try again.')}",
            "response_time": 0.0
        }

    severity_raw = state.get("severity", "unknown").upper()
    confidence = state.get("confidence", 0.0)
    faiss_results = state.get("graded_faiss_results") or state.get("faiss_results", [])
    retrieval_quality = state.get("retrieval_quality", "unknown")
    pubmed_papers = state.get("pubmed_results", [])
    web_sources = state.get("web_results", [])

    # Severity label with emoji
    severity_badges = {
        "MAJOR": "MAJOR 🔴",
        "MODERATE": "MODERATE 🟡",
        "MINOR": "MINOR 🟢",
    }
    severity_label = severity_badges.get(severity_raw, severity_raw)

    # FAISS confidence tier based on top score
    top_score = faiss_results[0].get("score", 0.0) if faiss_results else 0.0
    if top_score > 0.7:
        faiss_confidence = "high"
    elif top_score >= 0.5:
        faiss_confidence = "medium"
    else:
        faiss_confidence = "low"

    # Top FAISS interaction text for mechanism grounding
    top_faiss_text = faiss_results[0].get("interaction", "No local interaction text available.") if faiss_results else "No local interaction text available."

    from urllib.parse import quote_plus

    # FDA content — first 100 chars of actual content, not title
    fda_content = "No label found"
    for w in web_sources:
        raw = w.get("content", "").strip()
        if raw and raw != "FDA: No label found":
            fda_content = raw[:100]
            break

    # Pre-build PubMed lines with exact URLs — LLM must not touch these
    pubmed_source_lines = []
    pubmed_urls_list = []
    for p in pubmed_papers:
        title = p.get("title", "Unknown title")[:120]
        year = p.get("year", "N/A")
        url = p.get("url", "")
        pubmed_source_lines.append(f"- PubMed: {title} ({year}) → {url}")
        if url:
            pubmed_urls_list.append(url)

    if not pubmed_source_lines:
        pubmed_source_lines = ["- PubMed: No recent papers found"]

    # FDA search URL — constructed in Python, not by the LLM
    fda_url = f"https://labels.fda.gov/search?query={quote_plus(state['drug_1_clinical'])}"

    # Complete SOURCES block — built here so the LLM copies it verbatim
    sources_block = (
        f"- Local Database: {len(faiss_results)} matches - {faiss_confidence} confidence\n"
        + "\n".join(pubmed_source_lines)
        + f"\n- FDA: {fda_content}\n  → {fda_url}"
    )

    system = """You are a clinical decision support assistant.
Your job is to explain drug interactions at a pharmacological level.
Always explain the biological mechanism specifically — receptor binding, enzyme inhibition, metabolic pathways, pharmacokinetic effects.
Never say "consult a doctor" as the only recommendation — always include at least one specific clinical action.
Severity must exactly match what you are given. Output only the structured report, no preamble, no closing remarks."""

    prompt = f"""Generate a drug interaction report using the data below.

Drug 1: {state['drug_1_user']} (clinical: {state['drug_1_clinical']})
Drug 2: {state['drug_2_user']} (clinical: {state['drug_2_clinical']})
Severity (from assessment): {severity_raw}
Confidence score: {confidence:.2f}
Retrieval Quality: {retrieval_quality}

Top FAISS match (use this in MECHANISM):
{top_faiss_text}

Use EXACTLY these PubMed URLs, do not modify them:
{chr(10).join(pubmed_urls_list) if pubmed_urls_list else "None"}

Output this structure EXACTLY — no extra text, no deviation:

SEVERITY: {severity_label}

DRUGS: {state['drug_1_user']} + {state['drug_2_user']}

MECHANISM:
[2-3 sentences explaining WHY this interaction happens at a biological/pharmacological level. Be specific. Reference the top FAISS result above. Mention enzymes, receptors, or pathways by name.]

WHAT TO DO:
- [Specific clinical action 1]
- [Specific clinical action 2]
- [Specific clinical action 3]

SOURCES:
{sources_block}

CONFIDENCE: {int(confidence * 100)}%"""

    response = call_llm(prompt, system)
    response_time = time.time() - start_time

    return {
        "final_response": response,
        "response_time": response_time
    }