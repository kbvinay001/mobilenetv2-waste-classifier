"""home.py — GarbageSort AI Home Page (module card navigation)."""

import streamlit as st
import streamlit.components.v1 as components

MODULES = [
    dict(id="classify",  emoji="🗑️",  title="Classify Image",
         desc="Upload any waste image for instant MobileNetV2 classification with Grad-CAM visual explanation and recycling tips.",
         acc="acc-green",  color="#22C55E"),
    dict(id="live",      emoji="📷",  title="Live Camera",
         desc="Real-time waste classification from webcam or RTSP IP camera — optimised for edge devices at 1–5 FPS.",
         acc="acc-cyan",   color="#22D3EE"),
    dict(id="batch",     emoji="📦",  title="Batch Scan",
         desc="Upload a ZIP or multiple images (up to 50). Classify all at once and download results as CSV or PDF.",
         acc="acc-orange", color="#F97316"),
    dict(id="analytics", emoji="📊",  title="Session Analytics",
         desc="Confidence histogram, class distribution pie, and per-session timeline of all classifications made.",
         acc="acc-purple", color="#A78BFA"),
    dict(id="compare",   emoji="⚖️",  title="Model Compare",
         desc="Side-by-side CNN vs MobileNetV2 radar chart, per-class F1 scores, model size, and inference speed.",
         acc="acc-blue",   color="#3B82F6"),
    dict(id="system",    emoji="🔧",  title="System Info",
         desc="Edge model status, TFLite INT8 model details, CPU/RAM usage, and Raspberry Pi deployment guide.",
         acc="acc-yellow", color="#FBBF24"),
]


def show(go_page, session_stats: dict):
    total_cls  = session_stats.get("total_classified", 0)
    top_class  = session_stats.get("top_class", "—")
    avg_conf   = session_stats.get("avg_confidence", 0.0)
    model_name = session_stats.get("model_name", "MobileNetV2")

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="gs-hero">
  <div class="gs-welcome-badge">
    <span class="pulse-dot"></span>System Online &nbsp;—&nbsp; {model_name} loaded
  </div>
  <h1 style="
    font-family:'Playfair Display',Georgia,serif;
    font-size:clamp(3.2rem,6.5vw,5.2rem);
    font-weight:800; line-height:1.05;
    letter-spacing:-0.035em; color:#ffffff;
    margin:0 0 1rem 0;
    animation:fadeSlideUp .6s .1s ease both;
  ">
    Smart Waste<br>
    <span style="background:linear-gradient(135deg,#22C55E,#22D3EE);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
      Classification AI
    </span>
  </h1>
  <p class="gs-hero-sub">
    MobileNetV2 transfer learning classifies 7 waste categories with 93% accuracy.
    Grad-CAM explainability, live camera mode, and edge-device TFLite deployment — all in one dashboard.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Stat row ──────────────────────────────────────────────────────────
    conf_color = "#4ADE80" if avg_conf >= 70 else ("#FBBF24" if avg_conf >= 50 else "#F87171")
    st.markdown(f"""
<div class="gs-wrap">
  <div class="gs-stats-row">
    <div class="gs-stat-card" style="animation-delay:.00s">
      <div class="gs-stat-icon">🗑️</div>
      <div class="gs-stat-value">{total_cls:,}</div>
      <div class="gs-stat-label">Images Classified</div>
    </div>
    <div class="gs-stat-card" style="animation-delay:.08s">
      <div class="gs-stat-icon">🎯</div>
      <div class="gs-stat-value" style="color:{conf_color};">{avg_conf:.1f}%</div>
      <div class="gs-stat-label">Avg Confidence</div>
    </div>
    <div class="gs-stat-card" style="animation-delay:.16s">
      <div class="gs-stat-icon">🏆</div>
      <div class="gs-stat-value" style="font-size:1.15rem;">{top_class}</div>
      <div class="gs-stat-label">Top Class (Session)</div>
    </div>
    <div class="gs-stat-card" style="animation-delay:.24s">
      <div class="gs-stat-icon">🤖</div>
      <div class="gs-stat-value" style="font-size:1.1rem;">93%</div>
      <div class="gs-stat-label">Model Test Accuracy</div>
    </div>
  </div>
  <h2 class="gs-section-title">Explore Modules</h2>
</div>
""", unsafe_allow_html=True)

    # ── Module cards ──────────────────────────────────────────────────────
    def _card_row(mods, row_idx):
        cols = st.columns(3, gap="large")
        for i, mod in enumerate(mods):
            delay = (row_idx * 3 + i) * 0.08
            with cols[i]:
                st.markdown(f"""
<div class="mod-card-visual {mod['acc']}" style="animation-delay:{delay:.2f}s;">
  <div class="mod-icon-wrap">{mod['emoji']}</div>
  <div class="mod-card-title">{mod['title']}</div>
  <div class="mod-card-desc">{mod['desc']}</div>
</div>""", unsafe_allow_html=True)
                st.markdown(f'<div class="mod-explore {mod["acc"]}">', unsafe_allow_html=True)
                if st.button(f"Open {mod['title']} →", key=f"mod_{mod['id']}", use_container_width=True):
                    go_page(mod["id"])
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="max-width:1280px;margin:0 auto;padding:0 2.5rem">', unsafe_allow_html=True)
    _card_row(MODULES[:3], 0)
    st.markdown("<br>", unsafe_allow_html=True)
    _card_row(MODULES[3:], 1)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Info strip ────────────────────────────────────────────────────────
    st.markdown("""
<div class="gs-wrap" style="margin-top:1rem">
  <div class="gs-box eco">
    <strong style="color:#4ADE80;">♻  Edge-Device Ready</strong><br>
    <span style="color:var(--muted);font-size:.88rem;">
      TFLite INT8 quantized model (~7 MB) available for Raspberry Pi 4 / Jetson Nano deployment.
      Run <code style="background:rgba(255,255,255,.06);padding:1px 6px;border-radius:4px;">python edge/export_tflite.py</code>
      to generate the edge model, then check the <strong>System Info</strong> page.
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("""
<div class="gs-footer">
  Built with <span>TensorFlow · MobileNetV2 · Streamlit · Plotly</span> &nbsp;·&nbsp;
  4th Year Project &nbsp;·&nbsp; Smart Waste Classification AI
</div>
""", unsafe_allow_html=True)
