"""main.py — GarbageSort AI Streamlit entry point.

Launch:  streamlit run app/main.py
         (from the d:\\Vcodez_project root)
"""

import streamlit as st
import datetime
import sys
import uuid
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # d:\Vcodez_project
APP  = Path(__file__).parent                 # d:\Vcodez_project\app
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT))

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="GarbageSort AI — Smart Waste Classification",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Internal imports ──────────────────────────────────────────────────────────
from ui_styles import get_css, CONSTELLATION_JS

# ── CSS + particle canvas ─────────────────────────────────────────────────────
st.markdown(get_css(), unsafe_allow_html=True)
st.html(CONSTELLATION_JS)

# ── Session state defaults ────────────────────────────────────────────────────
_defaults = dict(
    page="home",
    model=None,
    session_id=str(uuid.uuid4())[:8].upper(),
    classification_log=[],
    live_active=False,
    live_queue=None,
    live_stop=None,
    live_session_log=[],
    system_log=[],
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Logging helper ────────────────────────────────────────────────────────────
def add_log(msg: str, level: str = "INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.system_log.append({"time": ts, "level": level, "msg": msg})
    if len(st.session_state.system_log) > 400:
        st.session_state.system_log = st.session_state.system_log[-400:]


def go_page(p: str):
    st.session_state.page = p
    st.rerun()


# ── Model loader (cached, loads once) ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from model_utils import load_model_compat
    model_path = ROOT / "models" / "transfer_learning_best.h5"
    if not model_path.exists():
        return None, f"Model not found: {model_path}"
    return load_model_compat(str(model_path))


# Load on first run
if st.session_state.model is None:
    with st.spinner("🤖 Loading MobileNetV2 model…"):
        _model, _err = load_model()
        if _err:
            add_log(f"Model load error: {_err}", "ALERT")
            st.error(f"❌ Failed to load model: {_err}")
            st.info("Ensure `models/transfer_learning_best.h5` exists in the project root.")
            st.stop()
        st.session_state.model = _model
        add_log("MobileNetV2 loaded  |  93.43% test accuracy  |  7-class waste classifier", "SUCCESS")
        add_log(f"Session {st.session_state.session_id} started", "INFO")

model      = st.session_state.model
session_id = st.session_state.session_id
page       = st.session_state.page

# ── Derived session stats ──────────────────────────────────────────────────
log = st.session_state.classification_log
_total   = len(log)
_avg_conf = sum(r["confidence"] for r in log) / _total if _total else 0.0
from collections import Counter
_counts = Counter(r["class"] for r in log)
_top    = _counts.most_common(1)[0][0] if _counts else "—"
session_stats = {
    "total_classified": _total,
    "avg_confidence":   _avg_conf,
    "top_class":        _top,
    "model_name":       "MobileNetV2",
}

# ── Navbar ────────────────────────────────────────────────────────────────────
model_tag = f"MobileNetV2 ✓" if model else "No model"
st.markdown(f"""
<div class="gs-nav">
  <div class="gs-nav-inner">
    <div class="gs-logo">
      <div class="gs-logo-icon">🗑️</div>
      GarbageSort AI
      <span class="gs-badge">v1.0</span>
    </div>
    <div class="gs-nav-right">
      <span class="pill-on">● {model_tag}</span>
      <span style="color:var(--muted);font-size:.8rem;">
        Session&nbsp;<strong style="color:#22C55E;">{session_id}</strong>
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Back button (all pages except home) ───────────────────────────────────────
if page != "home":
    st.markdown('<div class="gs-backrow">', unsafe_allow_html=True)
    if st.button("← Home", key="back_home"):
        go_page("home")
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if page == "home":
    from pages.home import show as show_home
    show_home(go_page, session_stats)

elif page == "classify":
    from pages.classify import show as show_classify
    show_classify(model, add_log, session_id)

elif page == "live":
    from pages.live_camera import show as show_live
    show_live(model, add_log, session_id)

elif page == "batch":
    from pages.batch import show as show_batch
    show_batch(model, add_log, session_id)

elif page == "analytics":
    from pages.analytics import show as show_analytics
    show_analytics(session_id)

elif page == "compare":
    from pages.model_compare import show as show_compare
    show_compare()

elif page == "system":
    from pages.system_info import show as show_system
    show_system()

else:
    go_page("home")

# ── System log expander (always at bottom) ────────────────────────────────────
if st.session_state.system_log:
    with st.expander("📋 System Log", expanded=False):
        log_lines = ""
        for entry in reversed(st.session_state.system_log[-50:]):
            lvl = entry["level"]
            cls = {"SUCCESS": "ls", "ALERT": "la", "WARN": "lw", "INFO": "li"}.get(lvl, "lt")
            log_lines += f'<span class="lt">[{entry["time"]}]</span> <span class="{cls}">[{lvl}]</span> {entry["msg"]}<br>'
        st.markdown(f'<div class="gs-log">{log_lines}</div>', unsafe_allow_html=True)
