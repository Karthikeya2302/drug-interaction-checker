# 💊 Drug Interaction Checker

> AI-powered clinical decision support system for real-time drug interaction analysis — built with Corrective RAG, LangGraph, and live biomedical data sources.
---

## What It Does

Type two drug names in plain English and get a **severity-rated clinical response** backed by three independent sources:

- 🗄️ **Local FAISS database** — 191,252 known drug interactions via semantic vector search
- 📚 **PubMed** — latest research papers via NCBI Entrez API with direct citation links
- 🏥 **FDA** — official drug labels via OpenFDA API with correct drug name resolution

### Example

**Input:** `Can I take Aspirin with Warfarin?`

**Output:**
```
SEVERITY: MAJOR 🔴
DRUGS: Warfarin + Aspirin

MECHANISM:
Warfarin inhibits vitamin K-dependent clotting factors while Aspirin irreversibly
inhibits COX-1, reducing thromboxane A2-mediated platelet aggregation. The combined
effect creates a synergistic anticoagulant action that significantly amplifies
bleeding risk.

WHAT TO DO:
• Monitor INR levels closely when initiating or discontinuing Aspirin therapy
• Adjust Warfarin dosage to maintain therapeutic INR range
• Watch for signs of bleeding: bruising, petechiae, GI bleeding

SOURCES:
• Local Database: 5 matches - high confidence
• PubMed: Drug-Nutrient Interactions with Acetylsalicylic acid and Warfarin (2024)
  → https://pubmed.ncbi.nlm.nih.gov/38612984/
• FDA: WARFARIN SODIUM — 7 DRUG INTERACTIONS
  → https://labels.fda.gov/search?query=Warfarin

CONFIDENCE: 90%
```

---

## Architecture

```
User Query (Natural Language)
         ↓
Node 1 — Drug Name Extraction + Normalization
         LLaMA 3.3 70B resolves brand names → clinical names
         (Aspirin → Acetylsalicylic acid, Tylenol → Acetaminophen)
         ↓
Node 2 ──────────────────── Node 3
FAISS Semantic Search       Live API Retrieval
191,252 interactions        PubMed Entrez + OpenFDA
Sentence Transformers       Parallel execution
         ↓                          ↓
         └────────────┬─────────────┘
                      ↓
         Node 4 — Severity Assessment
                  (major / moderate / minor)
                      ↓
         Node 5 — Structured Response Generation
                  LLaMA 3.3 70B with clinical prompt
                      ↓
         Streamlit UI + LangSmith Tracing
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq LLaMA 3.3 70B |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Vector DB | FAISS (191,252 interactions) |
| Workflow | LangGraph 5-node pipeline |
| RAG Strategy | Corrective RAG |
| Live Data | PubMed NCBI Entrez API + OpenFDA API |
| Observability | LangSmith — full node-level tracing |
| Frontend | Streamlit (clinical medical UI) |
| Dataset | 191,252 drug interactions (Kaggle) |

---

## Key Design Decisions

**Semantic Search over keyword matching** — FAISS finds relevant interactions even when users type brand names like `Aspirin` instead of the clinical name `Acetylsalicylic acid`, using cosine similarity on 384-dimension embeddings.

**Drug Normalization in Node 1** — LLaMA resolves synonyms, brand names, and common names before searching so queries always match the database correctly.

**Parallel Retrieval** — Nodes 2 and 3 run simultaneously, combining local and live evidence before severity assessment. This keeps average response time under 1.5s.

**Corrective RAG** — Every response is enriched with live PubMed research and FDA drug labels, not just static database results. If the local database has low-confidence matches, live sources compensate.

**LangSmith Observability** — Every query traces all 5 nodes in LangSmith: input/output per node, latency, confidence score, and source hit rates. This enables debugging retrieval quality at a per-node level.

**Structured Clinical Output** — Node 5 uses a strict system prompt enforcing SEVERITY → DRUGS → MECHANISM → WHAT TO DO → SOURCES → CONFIDENCE format, ensuring consistent, parseable responses.

---

## Project Structure

```
drug-interaction-checker/
├── src/
│   ├── rag/
│   │   ├── loader.py          # Load and clean CSV dataset
│   │   ├── embeddings.py      # Build FAISS index (~17 min, one-time)
│   │   └── retriever.py       # Semantic search with Sentence Transformers
│   ├── workflow/
│   │   ├── state.py           # LangGraph state definition
│   │   ├── nodes.py           # All 5 workflow nodes
│   │   └── graph.py           # Assemble full LangGraph pipeline
│   ├── monitoring/
│   │   └── tracker.py         # LangSmith interaction logging
│   └── evaluation/
│       └── benchmark.py       # Custom clinical evaluation framework
├── frontend/
│   └── app.py                 # Streamlit clinical UI
├── data/                      # Place CSV dataset here
├── config.py                  # Centralized configuration
├── requirements.txt
└── .env                       # API keys — never commit
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Karthikeya2302/drug-interaction-checker.git
cd drug-interaction-checker
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=drug-interaction-checker
```

Get your keys:
- **Groq API key** → [console.groq.com](https://console.groq.com)
- **LangSmith API key** → [smith.langchain.com](https://smith.langchain.com)

### 5. Download the dataset

Download from Kaggle: [mghobashy/drug-drug-interactions](https://www.kaggle.com/datasets/mghobashy/drug-drug-interactions)

Place it at: `data/db_drug_interactions.csv`

### 6. Build FAISS index

```bash
python -m src.rag.embeddings
```

⏱️ Takes ~17 minutes on CPU. One-time only — index is saved to `faiss_index/`.

### 7. Run the app

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 8. View LangSmith traces

Go to [smith.langchain.com](https://smith.langchain.com) → Projects → `drug-interaction-checker`

---

## Evaluation Results

| Metric | Value |
|--------|-------|
| Dataset size | 191,252 interactions |
| Average confidence | 87% |
| Average response time | 1.3s |
| Drug normalization | Brand → clinical name via LLM |
| Live sources per query | PubMed + FDA always retrieved |
| LangSmith traces | Full 5-node pipeline visibility |

---

## FDA Retrieval Fix

The OpenFDA API requires **lowercase** drug name queries against `openfda.generic_name`, `openfda.brand_name`, and `openfda.substance_name` fields in priority order. This project implements a 3-tier fallback strategy to ensure correct drug labels are always retrieved instead of unrelated results.

---

## Author

**Karthikeya Thimirishetty**  
MS Computer Science — University of Alabama at Birmingham (UAB)  

---

## Disclaimer

⚕️ This tool is for **informational and educational purposes only**. Always consult a licensed healthcare professional before making any medical decisions. This system is not a substitute for professional medical advice, diagnosis, or treatment.
