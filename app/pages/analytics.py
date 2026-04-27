"""analytics.py — Session analytics page for GarbageSort AI."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime, io

CLASS_LABELS  = ["Battery", "Cardboard", "Clothes", "Glass", "Metal", "Paper", "Plastic"]
CLASS_COLORS  = {
    "Battery": "#F87171", "Cardboard": "#FBBF24", "Clothes": "#A78BFA",
    "Glass": "#22D3EE", "Metal": "#60A5FA", "Paper": "#4ADE80", "Plastic": "#F97316",
}
CLASS_EMOJIS  = {
    "Battery": "🔋", "Cardboard": "📦", "Clothes": "👕",
    "Glass": "🥛", "Metal": "🥫", "Paper": "📄", "Plastic": "🧴",
}

_LAYOUT = dict(
    paper_bgcolor="#060913", plot_bgcolor="rgba(14,20,36,.4)",
    font=dict(family="Inter", color="#94A3B8"),
    xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8", size=10)),
)


def _chart(fig, h=280, title=""):
    fig.update_layout(
        height=h, margin=dict(t=36 if title else 24, b=20, l=30, r=16),
        title=dict(text=title, font=dict(size=11, color="#64748B")) if title else None,
        **_LAYOUT,
    )
    return fig


def show(session_id: str):
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
<div class="gs-hero">
  <div class="gs-welcome-badge"><span class="pulse-dot"></span>Analytics</div>
  <h1 class="gs-hero-title">
    Session <span class="text-gradient">Analytics</span>
  </h1>
  <p class="gs-hero-sub">
    Confidence distribution, class breakdown, timeline, and full per-image log for this session.
  </p>
</div>
<div class="gs-wrap">
""", unsafe_allow_html=True)

    log = st.session_state.get("classification_log", [])

    if not log:
        st.markdown("""
<div class="gs-box info" style="text-align:center;padding:3rem;">
  <div style="font-size:3rem;margin-bottom:1rem;">📊</div>
  <strong style="color:#60A5FA;">No data yet</strong><br>
  <span style="color:var(--muted);">Classify some images on the <strong>Classify Image</strong> or <strong>Batch Scan</strong> pages first.</span>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = pd.DataFrame(log)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Stats row ─────────────────────────────────────────────────────────
    avg_conf   = df["confidence"].mean()
    top_class  = df["class"].mode()[0] if len(df) else "—"
    conf_color = "#4ADE80" if avg_conf >= 70 else ("#FBBF24" if avg_conf >= 50 else "#F87171")
    low_conf   = (df["confidence"] < 50).sum()

    st.markdown(f"""
<div class="gs-stats-row">
  <div class="gs-stat-card">
    <div class="gs-stat-icon">🗑️</div>
    <div class="gs-stat-value">{len(df):,}</div>
    <div class="gs-stat-label">Total Classified</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">🎯</div>
    <div class="gs-stat-value" style="color:{conf_color};">{avg_conf:.1f}%</div>
    <div class="gs-stat-label">Avg Confidence</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">🏆</div>
    <div class="gs-stat-value" style="font-size:1.1rem;">{top_class}</div>
    <div class="gs-stat-label">Most Common Class</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">⚠️</div>
    <div class="gs-stat-value" style="color:#FBBF24;">{low_conf}</div>
    <div class="gs-stat-label">Low Confidence (&lt;50%)</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Row 1: Pie + Confidence histogram ─────────────────────────────────
    c1, c2 = st.columns(2, gap="large")

    with c1:
        vc = df["class"].value_counts().reset_index()
        vc.columns = ["class", "count"]
        colors = [CLASS_COLORS.get(c, "#22C55E") for c in vc["class"]]
        fig_pie = go.Figure(go.Pie(
            labels=[f"{CLASS_EMOJIS.get(c,'')} {c}" for c in vc["class"]],
            values=vc["count"],
            marker_colors=colors,
            hole=0.45,
            textinfo="label+percent",
            textfont=dict(size=11, color="#F1F5F9"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
        ))
        fig_pie.update_layout(
            height=300, margin=dict(t=36, b=10, l=10, r=10),
            paper_bgcolor="#060913",
            font=dict(family="Inter", color="#94A3B8"),
            title=dict(text="Class Distribution", font=dict(size=11, color="#64748B")),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#94A3B8")),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with c2:
        fig_hist = go.Figure(go.Histogram(
            x=df["confidence"],
            nbinsx=25,
            marker=dict(
                color=df["confidence"],
                colorscale=[[0, "#F87171"], [0.5, "#FBBF24"], [1, "#22C55E"]],
                showscale=False,
            ),
            opacity=0.85,
            hovertemplate="Confidence: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        ))
        _chart(fig_hist, 300, "Confidence Distribution")
        fig_hist.update_layout(
            xaxis_title="Confidence (%)", yaxis_title="Image Count",
        )
        # Add threshold lines
        for thresh, col, label in [(50, "#FBBF24", "50%"), (80, "#22C55E", "80%")]:
            fig_hist.add_vline(x=thresh, line_dash="dash", line_color=col,
                               annotation_text=label, annotation_font_color=col)
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # ── Row 2: Timeline + Per-class bar ───────────────────────────────────
    c3, c4 = st.columns(2, gap="large")

    with c3:
        # Timeline: confidence over time coloured by class
        fig_tl = go.Figure()
        for cls in df["class"].unique():
            sub = df[df["class"] == cls]
            fig_tl.add_trace(go.Scatter(
                x=sub.index, y=sub["confidence"],
                mode="markers+lines",
                name=f"{CLASS_EMOJIS.get(cls,'')} {cls}",
                marker=dict(color=CLASS_COLORS.get(cls, "#22C55E"), size=6),
                line=dict(color=CLASS_COLORS.get(cls, "#22C55E"), width=1.5),
                hovertemplate=f"<b>{cls}</b><br>Confidence: %{{y:.1f}}%<extra></extra>",
            ))
        _chart(fig_tl, 280, "Classification Timeline (Confidence by Image)")
        fig_tl.update_layout(
            xaxis_title="Image #", yaxis_title="Confidence (%)",
            yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,.04)"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9, color="#94A3B8")),
        )
        st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar": False})

    with c4:
        # Per-class average confidence
        cls_avg = df.groupby("class")["confidence"].mean().reset_index()
        cls_avg.columns = ["class", "avg_conf"]
        cls_avg["emoji"] = cls_avg["class"].map(CLASS_EMOJIS)
        colors2 = [CLASS_COLORS.get(c, "#22C55E") for c in cls_avg["class"]]
        fig_bar = go.Figure(go.Bar(
            x=[f"{e} {c}" for e, c in zip(cls_avg["emoji"], cls_avg["class"])],
            y=cls_avg["avg_conf"],
            marker_color=colors2,
            text=[f"{v:.1f}%" for v in cls_avg["avg_conf"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Avg Confidence: %{y:.1f}%<extra></extra>",
        ))
        _chart(fig_bar, 280, "Average Confidence per Class")
        fig_bar.update_layout(
            yaxis=dict(range=[0, 110], title="Avg Confidence (%)"),
            xaxis_title="",
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Session log table ──────────────────────────────────────────────────
    st.markdown('<h2 class="gs-section-title" style="margin-top:1rem;">Full Session Log</h2>',
                unsafe_allow_html=True)

    rows_html = ""
    for _, row in df.iterrows():
        c = CLASS_COLORS.get(row["class"], "#4ADE80")
        e = CLASS_EMOJIS.get(row["class"], "")
        ts = str(row["timestamp"])[:19]
        conf_bar = f'<div style="width:{row["confidence"]:.0f}%;height:4px;background:{c};border-radius:4px;"></div>'
        rows_html += f"""
<tr>
  <td style="color:var(--muted);font-size:.78rem;">{ts}</td>
  <td style="color:var(--txt2);font-size:.82rem;">{str(row.get("filename","—"))[:35]}</td>
  <td><span style="color:{c};font-weight:700;">{e} {row["class"]}</span></td>
  <td>
    <span style="color:{c};font-weight:700;">{row["confidence"]:.1f}%</span>
    {conf_bar}
  </td>
</tr>"""

    st.markdown(f"""
<div class="result-card" style="overflow-x:auto;max-height:360px;overflow-y:auto;">
<table class="gs-table">
  <tr><th>Timestamp</th><th>Filename</th><th>Class</th><th>Confidence</th></tr>
  {rows_html}
</table>
</div>""", unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────
    st.markdown('<h2 class="gs-section-title" style="margin-top:1.5rem;">Export</h2>',
                unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3, gap="large")
    with e1:
        csv = df.to_csv(index=False)
        st.download_button("⬇ Session CSV", csv,
                           f"session_{session_id}.csv", "text/csv",
                           use_container_width=True)
    with e2:
        import json
        st.download_button("⬇ Session JSON",
                           json.dumps(log, indent=2),
                           f"session_{session_id}.json", "application/json",
                           use_container_width=True)
    with e3:
        if st.button("🗑 Clear Session Log", use_container_width=True):
            st.session_state["classification_log"] = []
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
