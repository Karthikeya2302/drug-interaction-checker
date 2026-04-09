import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflow.graph import run_interaction_check
from src.monitoring.tracker import get_average_metrics

st.set_page_config(
    page_title="Drug Interaction Checker",
    page_icon="💊",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #080c14;
    --surface: #0d1424;
    --surface2: #111827;
    --border: #1e2d45;
    --accent: #00d4ff;
    --accent2: #0099cc;
    --major: #ff4444;
    --major-glow: rgba(255,68,68,0.3);
    --moderate: #ffaa00;
    --moderate-glow: rgba(255,170,0,0.3);
    --minor: #00cc66;
    --minor-glow: rgba(0,204,102,0.3);
    --text: #e2e8f0;
    --text-muted: #64748b;
    --text-dim: #334155;
}

/* Base */
.stApp {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 140px !important;
    max-width: 780px !important;
}

/* Title */
.app-title {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.5px;
    margin-bottom: 2px;
}
.app-subtitle {
    font-size: 12px;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 24px;
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--accent) 0%, transparent 100%);
    margin: 16px 0 28px 0;
    opacity: 0.4;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
    padding: 16px !important;
}

/* Severity badges */
.badge-major {
    display: inline-block;
    background: rgba(255,68,68,0.15);
    border: 1px solid var(--major);
    color: var(--major);
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    box-shadow: 0 0 12px var(--major-glow);
}
.badge-moderate {
    display: inline-block;
    background: rgba(255,170,0,0.15);
    border: 1px solid var(--moderate);
    color: var(--moderate);
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    box-shadow: 0 0 12px var(--moderate-glow);
}
.badge-minor {
    display: inline-block;
    background: rgba(0,204,102,0.15);
    border: 1px solid var(--minor);
    color: var(--minor);
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    box-shadow: 0 0 12px var(--minor-glow);
}
.badge-none, .badge-unknown {
    display: inline-block;
    background: rgba(100,116,139,0.15);
    border: 1px solid #475569;
    color: #94a3b8;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* Confidence bar */
.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 8px 0 16px 0;
}
.conf-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    min-width: 80px;
}
.conf-bar-bg {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    border-radius: 2px;
}
.conf-value {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    min-width: 36px;
    text-align: right;
}

/* Metrics row */
.metrics-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
}
.metric-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    flex: 1;
    text-align: center;
}
.metric-pill .m-label {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    display: block;
}
.metric-pill .m-value {
    font-family: 'Space Mono', monospace;
    font-size: 16px;
    color: var(--accent);
    font-weight: 700;
    display: block;
    margin-top: 2px;
}

/* Welcome card */
.welcome-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
}
.welcome-card h4 {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: var(--accent);
    margin: 0 0 12px 0;
    letter-spacing: 1px;
}
.example-query {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 8px;
    font-size: 13px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Footer */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 10px 32px;
    z-index: 999;
}
.footer-top {
    display: flex;
    gap: 24px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 4px;
}
.footer-tag {
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.5px;
}
.footer-tag b {
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 10px;
}
.footer-disclaimer {
    font-size: 10px;
    color: var(--text-dim);
    font-style: italic;
}

/* Streamlit metrics override */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 10px !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


def severity_badge(severity: str) -> str:
    s = severity.lower()
    labels = {
        "major": "⬤ MAJOR",
        "moderate": "⬤ MODERATE",
        "minor": "⬤ MINOR",
        "none": "⬤ NO INTERACTION",
        "unknown": "⬤ UNKNOWN"
    }
    label = labels.get(s, "⬤ UNKNOWN")
    return f'<span class="badge-{s}">{label}</span>'


def confidence_bar(conf: float) -> str:
    pct = int(conf * 100)
    width = pct
    return f"""
    <div class="conf-row">
        <span class="conf-label">Confidence</span>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{width}%"></div>
        </div>
        <span class="conf-value">{pct}%</span>
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
st.markdown('<div class="app-title">💊 Drug Interaction Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">RAG · LangGraph · LLaMA 3.3 70B · PubMed · FDA</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ── Welcome card — only show when no chat history ─────────
if not st.session_state.chat_history:
    st.markdown("""
<div class="welcome-card">
    <h4>// WHAT THIS DOES</h4>
    <p style="color:#94a3b8; font-size:13px; margin-bottom:16px;">
        Type two drug names in plain English. The system checks 191,252 known interactions, 
        searches live PubMed research and FDA warnings, then gives you a severity-rated 
        clinical response in seconds.
    </p>
    <h4>// EXAMPLE QUERIES</h4>
    <div class="example-query">Can I take Aspirin with Warfarin?</div>
    <div class="example-query">Is it safe to combine Metformin and Ibuprofen?</div>
    <div class="example-query">What happens if I take Lisinopril with Potassium?</div>
    <div class="example-query">Can I take Tylenol with Advil?</div>
</div>
""", unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["query"])
    with st.chat_message("assistant"):
        st.markdown(severity_badge(chat["severity"]), unsafe_allow_html=True)
        st.markdown(confidence_bar(chat["confidence"]), unsafe_allow_html=True)
        st.markdown(chat["response"])

# ── Chat input ────────────────────────────────────────────
user_input = st.chat_input("Ask about a drug combination...")

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
                st.markdown(result.get("final_response", ""))

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
    <div class="footer-top">
        <span class="footer-tag"><b>MODEL</b> LLaMA 3.3 70B via Groq</span>
        <span class="footer-tag"><b>RAG</b> FAISS + Sentence Transformers · 191,252 interactions</span>
        <span class="footer-tag"><b>WORKFLOW</b> LangGraph 5-node pipeline</span>
        <span class="footer-tag"><b>STATS</b> {total_q} queries · {avg_conf} avg confidence · {avg_time} avg response · {avg_faiss} avg FAISS</span>
    </div>
    <div class="footer-disclaimer">⚕️ This tool is for informational purposes only. Always consult a licensed healthcare professional before making medical decisions.</div>
</div>
""", unsafe_allow_html=True)