"""Streamlit UI for Concept2Paper MVP."""

import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("API_URL", "http://api:8000/api/v1")
POLL_INTERVAL = 2  # seconds


st.set_page_config(page_title="Concept2Paper", page_icon="📄", layout="wide")

st.title("📄 Concept2Paper")
st.markdown(
    "Select a research concept below and discover the **top-5 unexplored connections** "
    "predicted by a link-prediction model. Then generate a paper draft outline with AI."
)

# ── fetch available concepts ──────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_concepts():
    try:
        r = requests.get(f"{API_BASE}/concepts", timeout=10)
        r.raise_for_status()
        return r.json()["concepts"]
    except Exception as e:
        st.error(f"Could not load concepts from API: {e}")
        return []


concepts = fetch_concepts()

if not concepts:
    st.warning("No concepts available. Is the API running?")
    st.stop()

# ── sidebar: keyword selection ────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    keyword = st.selectbox("Select a concept keyword", concepts, index=0)
    find_btn = st.button("🔍 Find Research Opportunities", use_container_width=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def poll_session(session_id: str, timeout: int = 120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{API_BASE}/sessions/{session_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(POLL_INTERVAL)
    return None


def poll_draft(draft_id: str, timeout: int = 60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{API_BASE}/drafts/{draft_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(POLL_INTERVAL)
    return None


def create_session(keyword: str):
    r = requests.post(f"{API_BASE}/sessions", json={"keyword": keyword}, timeout=10)
    r.raise_for_status()
    return r.json()


def request_draft(concept_a: str, concept_b: str):
    r = requests.post(
        f"{API_BASE}/drafts", json={"concept_a": concept_a, "concept_b": concept_b}, timeout=10
    )
    r.raise_for_status()
    return r.json()


# ── main logic ────────────────────────────────────────────────────────────────

if find_btn:
    st.session_state.pop("session_result", None)
    st.session_state.pop("drafts", None)

    with st.spinner(f"Scoring all candidate pairs for **{keyword}**… (may take ~30 s)"):
        try:
            session = create_session(keyword)
            result = poll_session(session["id"])
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if result is None:
        st.error("Timed out waiting for inference results.")
        st.stop()
    if result["status"] == "error":
        st.error(f"Inference error: {result.get('error_msg', 'unknown')}")
        st.stop()

    st.session_state["session_result"] = result
    st.session_state["keyword"] = keyword
    st.session_state["drafts"] = {}


# ── display results ───────────────────────────────────────────────────────────

if "session_result" in st.session_state:
    result = st.session_state["session_result"]
    kw = st.session_state.get("keyword", "")
    pairs = result.get("pairs", [])

    st.subheader(f"Top {len(pairs)} predicted connections for **{kw}**")

    if not pairs:
        st.info("No candidate pairs found.")
    else:
        for pair in pairs:
            rank = pair["rank"]
            concept_a = pair["concept_a"]
            concept_b = pair["concept_b"]
            score = pair["score"]

            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(
                        f"**#{rank}** &nbsp; `{concept_a}` × `{concept_b}`"
                        f"  &nbsp; — &nbsp; score: **{score:.4f}**"
                    )
                with col2:
                    draft_key = f"draft_{rank}"
                    if st.button("✍️ Generate Draft", key=f"btn_{rank}"):
                        with st.spinner("Generating draft outline…"):
                            try:
                                draft_resp = request_draft(concept_a, concept_b)
                                draft_data = poll_draft(draft_resp["id"])
                                st.session_state["drafts"][draft_key] = draft_data
                            except Exception as e:
                                st.error(f"Draft error: {e}")

                # show draft if available
                if draft_key in st.session_state.get("drafts", {}):
                    draft = st.session_state["drafts"][draft_key]
                    if draft["status"] == "done":
                        with st.expander(f"📝 Draft for #{rank}", expanded=True):
                            st.markdown(draft["content"])
                    elif draft["status"] == "error":
                        st.error(f"Draft failed: {draft.get('error_msg')}")

                st.divider()
