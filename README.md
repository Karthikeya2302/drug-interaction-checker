# 💊 Drug Interaction Checker

**An AI system that tells you if two drugs are dangerous to take together — and explains exactly why, with real medical sources.**

Type two drug names in plain English. Get a severity rating, biological explanation, and citations from PubMed and FDA in under 2 seconds.

> Built with Corrective RAG · LangGraph · LLaMA 3.3 70B · LangSmith

---

## The Problem It Solves

Most drug interaction checkers just say **"Warning: Major Interaction."** They don't explain why, don't cite sources, and can't handle brand names like "Tylenol" or "Advil."

This system does all three:

| Traditional Checker | This System |
|---------------------|-------------|
| ⚠️ "Major interaction detected" | 🔴 MAJOR — explains the biological mechanism |
| No sources | Cites real PubMed papers + FDA drug labels |
| Fails on brand names | Resolves Aspirin → Acetylsalicylic acid automatically |
| Static database lookup | Live retrieval from PubMed + FDA every query |
| No quality check on results | Grades retrieval quality, discards irrelevant results |

---

## Demo

**Input:** `Can I take Aspirin with Warfarin?`

```
SEVERITY: MAJOR 🔴
DRUGS: Warfarin + Aspirin

MECHANISM:
Warfarin inhibits vitamin K-dependent clotting factors (II, VII, IX, X).
Aspirin irreversibly blocks COX-1, reducing platelet aggregation.
Together, they create a synergistic anticoagulant effect that
dramatically increases bleeding risk beyond either drug alone.

WHAT TO DO:
• Monitor INR levels closely when starting or stopping Aspirin
• Adjust Warfarin dose to maintain therapeutic INR range
• Watch for bleeding signs: bruising, petechiae, GI bleeding

SOURCES:
• Local Database: 5 matches — high confidence
• PubMed: Drug-Nutrient Interactions with Warfarin (2024)
  → https://pubmed.ncbi.nlm.nih.gov/38612984/
• FDA: WARFARIN SODIUM — Drug Interactions section
  → https://labels.fda.gov/search?query=Warfarin

CONFIDENCE: 90%
```

---

## How It Works

The system runs a **6-node LangGraph pipeline** on every query:

```
Your Query
    │
    ▼
[Node 1] Drug Name Extraction & Normalization
         LLaMA 3.3 70B converts brand names to clinical names
         "Aspirin" → "Acetylsalicylic acid"
         "Tylenol" → "Acetaminophen"
    │
    ├──────────────────────┐
    ▼                      ▼
[Node 2]              [Node 3]
FAISS Search          Live API Search
191,252 interactions  PubMed + FDA
runs in parallel      runs in parallel
    │                      │
    ▼                      │
[Node 2.5] ◄───────────────┘
Retrieval Grader  ← THIS is what makes it Corrective RAG
  score ≥ 0.7  → relevant ✅ keep
  score 0.5–0.7 → partial ⚠️ keep but flagged
  score < 0.5  → noise ❌ discard
    │
    ▼
[Node 4] Severity Assessment
         major / moderate / minor
    │
    ▼
[Node 5] Response Generation
         LLaMA 3.3 70B with clinical system prompt
         Structured output with sources + confidence
    │
    ▼
Streamlit UI  +  LangSmith Tracing
```

### Why Corrective RAG matters here

Standard RAG retrieves and immediately generates — even if the retrieved results are irrelevant. Node 2.5 fixes this by **grading every FAISS result** before the LLM sees it. Results below 0.5 cosine similarity are discarded as noise. If no results pass the threshold, the system flags `retrieval_corrected = True` and relies on live PubMed + FDA sources instead.

---

## Tech Stack

| What | How |
|------|-----|
| LLM | LLaMA 3.3 70B via Groq API |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Vector DB | FAISS — 191,252 drug interactions |
| Pipeline | LangGraph 6-node workflow |
| RAG Strategy | Corrective RAG with retrieval grading |
| Live Medical Data | PubMed NCBI Entrez API + OpenFDA API |
| Observability | LangSmith — traces every node |
| Frontend | Streamlit |
| Dataset | 191,252 DDI pairs (Kaggle) |

---

## Performance

| Metric | Value |
|--------|-------|
| Dataset coverage | 191,252 drug interactions |
| Avg confidence score | 87% |
| Avg response time | ~1.3 seconds |
| Drug normalization | Brand names → clinical names via LLM |
| Sources per query | FAISS + PubMed + FDA (always 3) |
| Pipeline observability | Full 6-node LangSmith traces |

---

## Setup

**Requirements:** Python 3.11+, Groq API key, LangSmith API key

```bash
# 1. Clone
git clone https://github.com/Karthikeya2302/drug-interaction-checker.git
cd drug-interaction-checker

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install
pip install -r requirements.txt

# 4. Environment variables — create .env file
GROQ_API_KEY=your_groq_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=drug-interaction-checker

# 5. Download dataset from Kaggle → place at data/db_drug_interactions.csv
# https://www.kaggle.com/datasets/mghobashy/drug-drug-interactions

# 6. Build FAISS index (one-time, ~17 minutes)
python -m src.rag.embeddings

# 7. Run
streamlit run frontend/app.py
```

Get API keys: [Groq](https://console.groq.com) · [LangSmith](https://smith.langchain.com)

---

## Project Structure

```
drug-interaction-checker/
├── src/
│   ├── rag/
│   │   ├── loader.py        # Dataset loading + cleaning
│   │   ├── embeddings.py    # FAISS index builder
│   │   └── retriever.py     # Semantic search
│   ├── workflow/
│   │   ├── state.py         # LangGraph state schema
│   │   ├── nodes.py         # All 6 pipeline nodes
│   │   └── graph.py         # Pipeline assembly
│   └── monitoring/
│       └── tracker.py       # LangSmith logging
├── frontend/
│   └── app.py               # Streamlit UI
├── config.py
├── requirements.txt
└── .env                     # Never commit this
```

---

## Author

**Karthikeya Thimirishetty**
MS Computer Science · University of Alabama at Birmingham (UAB)

---

*⚕️ For educational purposes only. Always consult a licensed healthcare professional before making medical decisions.*
