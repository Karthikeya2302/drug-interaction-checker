from dotenv import load_dotenv
load_dotenv()

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "drug-interaction-checker"

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflow.graph import run_interaction_check
from src.monitoring.tracker import get_average_metrics

st.set_page_config(
    page_title="Drug Interaction Checker",
    page_icon="✚",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

:root {
    --bg: #f8fafc;
    --surface: #ffffff;
    --surface2: #f1f5f9;
    --border: #e2e8f0;
    --accent: #0066cc;
    --accent-light: #e8f0fe;
    --major: #dc2626;
    --major-bg: #fef2f2;
    --moderate: #d97706;
    --moderate-bg: #fffbeb;
    --minor: #16a34a;
    --minor-bg: #f0fdf4;
    --text: #0f172a;
    --text-muted: #64748b;
}

/* ── Base ── */
.stApp {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 130px !important;
    max-width: 800px !important;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 22px 0 18px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.app-header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.app-cross {
    font-size: 24px;
    color: var(--major);
    line-height: 1;
}
.app-title {
    font-family: 'Instrument Serif', serif;
    font-size: 26px;
    font-weight: 400;
    color: var(--text);
    letter-spacing: -0.2px;
    line-height: 1;
}
.app-badge {
    background: var(--accent-light);
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 20px;
    letter-spacing: 0.2px;
}

/* ── Welcome card ── */
.welcome-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 28px 28px 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.welcome-heading {
    font-family: 'Instrument Serif', serif;
    font-size: 22px;
    color: var(--text);
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.welcome-sub {
    font-size: 13px;
    color: var(--text-muted);
    margin: 0 0 20px 0;
    line-height: 1.6;
}
.stat-pills {
    display: flex;
    gap: 8px;
    margin-bottom: 22px;
    flex-wrap: wrap;
}
.stat-pill {
    background: var(--accent-light);
    color: var(--accent);
    font-size: 12px;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 20px;
}
.examples-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
}
.example-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.example-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    color: var(--text-muted);
    cursor: default;
    transition: border-color 0.15s, color 0.15s, background 0.15s;
}
.example-chip:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-light);
}

/* ── Severity badges ── */
.badge-major {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--major-bg);
    border: 1px solid var(--major);
    color: var(--major);
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.4px;
}
.badge-moderate {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--moderate-bg);
    border: 1px solid var(--moderate);
    color: var(--moderate);
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.4px;
}
.badge-minor {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--minor-bg);
    border: 1px solid var(--minor);
    color: var(--minor);
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.4px;
}
.badge-none, .badge-unknown {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface2);
    border: 1px solid #cbd5e1;
    color: var(--text-muted);
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.4px;
}

/* ── Confidence bar ── */
.conf-wrapper {
    margin: 12px 0 18px 0;
}
.conf-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.conf-label {
    font-size: 10px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.conf-value {
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
}
.conf-bar-bg {
    width: 100%;
    height: 8px;
    background: var(--surface2);
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.conf-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #0066cc, #60a5fa);
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* ── Metrics row ── */
.metrics-row {
    display: flex;
    gap: 10px;
    margin-top: 16px;
}
.metric-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    flex: 1;
    text-align: center;
}
.metric-pill .m-label {
    font-size: 10px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    display: block;
}
.metric-pill .m-value {
    font-size: 18px;
    font-weight: 700;
    color: var(--accent);
    display: block;
    margin-top: 3px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
    padding: 18px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

/* ── Chat message text ── */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] strong {
    color: #0f172a !important;
}

/* ── Chat input ── */
[data-testid="stBottom"] {
    background: #ffffff !important;
    border-top: 1px solid #e2e8f0 !important;
    box-shadow: 0 -4px 12px rgba(0,0,0,0.06) !important;
}
section[data-testid="stBottom"] > div {
    background: #ffffff !important;
}
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    background: #ffffff !important;
}

/* ── Footer ── */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--bg);
    border-top: 1px solid var(--border);
    padding: 10px 32px 10px 32px;
    z-index: 999;
}
.footer-stats {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.footer-stats span b {
    color: var(--accent);
    font-weight: 600;
}
.footer-disclaimer {
    font-size: 11px;
    color: #94a3b8;
    font-style: italic;
}

/* ── Streamlit metric overrides ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 10px !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


def severity_badge(severity: str) -> str:
    s = severity.lower()
    labels = {
        "major": "● MAJOR",
        "moderate": "● MODERATE",
        "minor": "● MINOR",
        "none": "● NO INTERACTION",
        "unknown": "● UNKNOWN"
    }
    label = labels.get(s, "● UNKNOWN")
    return f'<span class="badge-{s}">{label}</span>'


def confidence_bar(conf: float) -> str:
    pct = int(conf * 100)
    return f"""
    <div class="conf-wrapper">
        <div class="conf-label-row">
            <span class="conf-label">Confidence Score</span>
            <span class="conf-value">{pct}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct}%"></div>
        </div>
    </div>
    """


def metrics_html(response_time: float, faiss_count: int, live_sources: int) -> str:
    return f"""
    <div class="metrics-row">
        <div class="metric-pill">
            <span class="m-label">Response</span>
            <span class="m-value">{response_time:.1f}s</span>
        </div>
        <div class="metric-pill">
            <span class="m-label">FAISS Matches</span>
            <span class="m-value">{faiss_count}</span>
        </div>
        <div class="metric-pill">
            <span class="m-label">Live Sources</span>
            <span class="m-value">{live_sources}</span>
        </div>
    </div>
    """


# ── Session state ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-header-left">
        <span class="app-cross">✚</span>
        <span class="app-title">Drug Interaction Checker</span>
    </div>
    <span class="app-badge">Clinical Decision Support</span>
</div>
""", unsafe_allow_html=True)

# ── Welcome card — only show when no chat history ─────────
if not st.session_state.chat_history:
    st.markdown("""
<div class="welcome-card">
    <div class="welcome-heading">Check Drug Interactions</div>
    <div class="welcome-sub">
        Powered by 191,252 clinical interactions · Live PubMed · FDA Drug Labels
    </div>
    <div class="stat-pills">
        <span class="stat-pill">191K Interactions</span>
        <span class="stat-pill">Live PubMed</span>
        <span class="stat-pill">FDA Labels</span>
    </div>
    <div class="examples-label">Example Queries</div>
    <div class="example-chips">
        <span class="example-chip">Can I take Aspirin with Warfarin?</span>
        <span class="example-chip">Is Metformin safe with Alcohol?</span>
        <span class="example-chip">Sertraline and Tramadol interaction?</span>
        <span class="example-chip">Can I take Lisinopril with Potassium?</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["query"])
    with st.chat_message("assistant"):
        st.markdown(severity_badge(chat["severity"]), unsafe_allow_html=True)
        st.markdown(confidence_bar(chat["confidence"]), unsafe_allow_html=True)
        st.markdown(chat["response"], unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────
user_input = st.chat_input("e.g. Can I take Warfarin with Aspirin?")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing across local database, PubMed and FDA..."):
            try:
                result = run_interaction_check(user_input.strip())
                severity = result.get("severity", "unknown")
                confidence = result.get("confidence", 0.0)
                response_time = result.get("response_time", 0.0)
                faiss_count = len(result.get("faiss_results", []))
                live_sources = (
                    len(result.get("pubmed_results", [])) +
                    len(result.get("web_results", []))
                )

                st.markdown(severity_badge(severity), unsafe_allow_html=True)
                st.markdown(confidence_bar(confidence), unsafe_allow_html=True)
                st.markdown(result.get("final_response", ""), unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "query": user_input.strip(),
                    "response": result.get("final_response", ""),
                    "severity": severity,
                    "confidence": confidence,
                    "response_time": response_time,
                    "faiss_count": faiss_count,
                    "live_sources": live_sources
                })

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

# ── Footer ────────────────────────────────────────────────
metrics = get_average_metrics()
avg_conf = f"{metrics.get('avg_confidence', 0):.0%}" if metrics else "—"
avg_time = f"{metrics.get('avg_response_time', 0):.1f}s" if metrics else "—"
avg_faiss = f"{metrics.get('avg_faiss_score', 0):.2f}" if metrics else "—"
total_q = metrics.get("total_queries", 0) if metrics else 0

st.markdown(f"""
<div class="footer">
    <div class="footer-stats">
        <span><b>Model</b> LLaMA 3.3 70B via Groq</span>
        <span><b>Dataset</b> 191,252 interactions · FAISS + Sentence Transformers</span>
        <span><b>Pipeline</b> LangGraph 5-node</span>
        <span><b>Stats</b> {total_q} queries · {avg_conf} avg confidence · {avg_time} avg response</span>
    </div>
    <div class="footer-disclaimer">⚕ For informational purposes only. Always consult a licensed healthcare professional before making medical decisions.</div>
</div>
""", unsafe_allow_html=True)
