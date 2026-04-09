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


def search_fda(drug_1: str, drug_2: str) -> list:
    """Search FDA drug label database — free, no key required."""
    import requests

    try:
        response = requests.get(
            "https://api.fda.gov/drug/label.json",
            params={
                "search": f"drug_interactions:{drug_1} {drug_2}",
                "limit": 3
            },
            timeout=10
        )
        data = response.json()

        results = []
        for item in data.get("results", []):
            brand_name = item.get("openfda", {}).get("brand_name", ["Unknown"])[0]
            interaction_text = item.get("drug_interactions", [""])[0]
            results.append({
                "title": f"FDA Label: {brand_name}",
                "content": interaction_text[:300],
                "source": "FDA",
                "url": "https://fda.gov"
            })

        return results

    except Exception as e:
        print(f"  FDA search failed: {e}")
        return []

# ── Node 4 ────────────────────────────────────────────────────────────────
def assess_severity(state: DrugInteractionState) -> dict:
    """
    Combine FAISS + MCP results.
    Classify interaction severity using LLaMA.
    """
    print("Node 4: Assessing severity...")

    if not state.get("drugs_validated"):
        return {"severity": "unknown", "confidence": 0.0}

    # Combine all evidence
    faiss_text = "\n".join([r["interaction"] for r in state.get("faiss_results", [])])
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
    """
    Generate final human readable response.
    Clean format — no disclaimer, no confidence in body.
    """
    print("Node 5: Generating final response...")

    start_time = time.time()

    # Handle validation failure
    if not state.get("drugs_validated"):
        return {
            "final_response": f"❌ {state.get('validation_error', 'Could not identify the drugs. Please try again.')}",
            "response_time": 0.0
        }

    severity = state.get("severity", "unknown").upper()
    faiss_count = len(state.get("faiss_results", []))
    pubmed_papers = state.get("pubmed_results", [])
    web_sources = state.get("web_results", [])

    # Format pubmed sources cleanly
    pubmed_lines = "\n".join([
        f"• {p['title'][:80]}... ({p.get('year', 'N/A')}) — {p['url']}"
        for p in pubmed_papers
    ]) or "• No recent papers found"

    # Format FDA sources cleanly
    fda_lines = "\n".join([
        f"• {w.get('title', '')[:80]}"
        for w in web_sources
    ]) or "• No FDA results found"

    # Format local DB findings
    local_findings = "\n".join([
        r['interaction'][:150]
        for r in state.get('faiss_results', [])
    ])

    system = """You are a clinical pharmacist explaining drug interactions clearly.
Be precise and helpful. Never give a final medical decision."""

    prompt = f"""Generate a drug interaction report using this data:

Drug 1: {state['drug_1_user']} (clinical: {state['drug_1_clinical']})
Drug 2: {state['drug_2_user']} (clinical: {state['drug_2_clinical']})
Severity: {severity}

Local Database ({faiss_count} matches):
{local_findings}

PubMed Research:
{pubmed_lines}

FDA Sources:
{fda_lines}

Format your response EXACTLY like this — no extra text, no deviation, no disclaimer at end:

**Drugs:** {state['drug_1_user']} ({state['drug_1_clinical']}) + {state['drug_2_user']} ({state['drug_2_clinical']})

**What happens:**
[2-3 sentences explaining the interaction in plain English]

**Recommendation:**
- [action 1]
- [action 2]
- [action 3]

**Sources:**
- Local Drug Database — {faiss_count} matches
{pubmed_lines}
{fda_lines}"""

    response = call_llm(prompt, system)
    response_time = time.time() - start_time

    return {
        "final_response": response,
        "response_time": response_time
    }