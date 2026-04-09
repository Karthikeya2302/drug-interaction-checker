# 💊 Drug Interaction Checker

An AI-powered drug interaction checker built with Corrective RAG, LangGraph, and live biomedical data sources.

---

## What It Does

Type two drug names in plain English and get a severity-rated clinical response backed by three independent sources:

- **Local database** — 191,252 known drug interactions via FAISS vector search
- **PubMed** — latest research papers via NCBI Entrez API
- **FDA** — official drug labels via FDA OpenAPI

---

## Example

**Input:** Can I take Aspirin with Warfarin?

**Output:**

🔴 MAJOR INTERACTION

**Drugs:** Aspirin (Acetylsalicylic acid) + Warfarin (Warfarin)

**What happens:**
Warfarin may increase the anticoagulant activities of Aspirin, leading to increased bleeding risk. Both medications have blood-thinning properties and combining them amplifies this effect.

**Recommendation:**
- Inform your doctor immediately
- Monitor for signs of bleeding
- Do not stop either medication without consulting your doctor

**Sources:**
- Local Drug Database — 5 matches
- PubMed: Bleeding risk with aspirin-warfarin combination (2024)
- FDA Label: Warfarin Sodium

---

## Architecture

```
User Query
    ↓
Node 1 — Extract + normalize drug names (LLaMA 3.3 70B)
    ↓
Node 2 ──────────────── Node 3
FAISS local search       Live API search
191,252 interactions     PubMed + FDA
    ↓                        ↓
    └─────────┬──────────────┘
              ↓
       Node 4 — Severity assessment
              ↓
       Node 5 — Generate response (LLaMA 3.3 70B)
              ↓
       Streamlit UI + MLflow tracking
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq LLaMA 3.3 70B |
| Embeddings | Sentence Transformers all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| Workflow | LangGraph 5-node pipeline |
| Live Data | PubMed Entrez API + FDA OpenAPI |
| Tracking | MLflow |
| Frontend | Streamlit |
| Dataset | 191,252 drug interactions (Kaggle) |

---

## Project Structure

```
drug-interaction-checker/
├── src/
│   ├── rag/
│   │   ├── loader.py        # load and clean CSV
│   │   ├── embeddings.py    # build FAISS index
│   │   └── retriever.py     # semantic search
│   ├── workflow/
│   │   ├── state.py         # LangGraph state definition
│   │   ├── nodes.py         # all 5 workflow nodes
│   │   └── graph.py         # assemble full pipeline
│   └── monitoring/
│       └── tracker.py       # MLflow experiment logging
├── frontend/
│   └── app.py               # Streamlit chat UI
├── data/                    # place CSV here
├── config.py                # centralized settings
├── requirements.txt
└── .env                     # API keys — never commit
```

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/yourusername/drug-interaction-checker.git
cd drug-interaction-checker
```

**2. Create virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add your API key**

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
```

**5. Download the dataset**

Download from Kaggle: [mghobashy/drug-drug-interactions](https://www.kaggle.com/datasets/mghobashy/drug-drug-interactions)

Place it at:

```
data/db_drug_interactions.csv
```

**6. Build FAISS index** (one time only — takes ~17 minutes on CPU)

```bash
python -m src.rag.embeddings
```

**7. Run the app**

```bash
streamlit run frontend/app.py
```

**8. View MLflow dashboard**

```bash
mlflow ui
```

Open http://127.0.0.1:5000

---

## Key Design Decisions

**Semantic search** — FAISS finds relevant interactions even when users type brand names like Aspirin instead of the clinical name Acetylsalicylic acid.

**Drug normalization** — Node 1 uses LLaMA to resolve synonyms before searching so queries always match the database correctly.

**Parallel retrieval** — Nodes 2 and 3 run simultaneously, combining local and live evidence before assessment.

**Corrective RAG** — Every response is enriched with live PubMed research and FDA warnings, not just static database results.

**MLflow tracking** — Every query logs confidence score, FAISS similarity, response time, and source counts for evaluation.

---

## Evaluation Results

| Metric | Value |
|--------|-------|
| Dataset size | 191,252 interactions |
| Average confidence | 87% |
| Average response time | 1.3s |
| Synonym resolution | Brand to clinical name |
| Live sources per query | PubMed + FDA always |


---

## Author

**Karthikeya Thimirishetty**
MS Computer Science — University of Alabama at Birmingham (UAB)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](http://www.linkedin.com/in/karthikeya-thimirishetty-uab24)

---
## Demo

▶️ [Watch Demo Video](https://app.guidde.com/share/playbooks/uJ9NxCof12uUHyF3RNyuYv?origin=TCRYtPia46ZdEmANPTlDUTPwsxk1&mode=videoOnly)

---
## Disclaimer

⚕️ This tool is for informational purposes only. Always consult a licensed healthcare professional before making medical decisions.
