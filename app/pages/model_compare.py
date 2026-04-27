"""model_compare.py — CNN vs MobileNetV2 comparison page for GarbageSort AI."""

import streamlit as st
import plotly.graph_objects as go
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

CLASS_LABELS = ["Battery", "Cardboard", "Clothes", "Glass", "Metal", "Paper", "Plastic"]

# ── Ground-truth metrics from saved classification reports ─────────────────
CNN_F1 = {
    "Battery":   0.89, "Cardboard": 0.81, "Clothes": 0.81,
    "Glass":     0.76, "Metal":     0.70, "Paper":   0.83, "Plastic": 0.63,
}
CNN_ACCURACY    = 77.88
CNN_PRECISION   = 78.0
CNN_RECALL      = 77.0

MBV2_F1 = {
    "Battery":   0.98, "Cardboard": 0.92, "Clothes": 0.99,
    "Glass":     0.91, "Metal":     0.95, "Paper":   0.90, "Plastic": 0.88,
}
MBV2_ACCURACY   = 93.43
MBV2_PRECISION  = 93.6
MBV2_RECALL     = 93.6

_LAYOUT = dict(
    paper_bgcolor="#060913", plot_bgcolor="rgba(14,20,36,.4)",
    font=dict(family="Inter", color="#94A3B8"),
    margin=dict(t=40, b=20, l=20, r=20),
)


def show():
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
<div class="gs-hero">
  <div class="gs-welcome-badge"><span class="pulse-dot"></span>Model Analysis</div>
  <h1 class="gs-hero-title">
    Model <span class="text-gradient">Comparison</span>
  </h1>
  <p class="gs-hero-sub">
    CNN from scratch vs MobileNetV2 transfer learning.
    Radar chart, per-class F1 breakdown, accuracy bars, and deployment metrics.
  </p>
</div>
<div class="gs-wrap">
""", unsafe_allow_html=True)

    # ── Summary stat cards ─────────────────────────────────────────────────
    improvement = MBV2_ACCURACY - CNN_ACCURACY
    st.markdown(f"""
<div class="gs-stats-row">
  <div class="gs-stat-card">
    <div class="gs-stat-icon">🧠</div>
    <div class="gs-stat-value">{CNN_ACCURACY:.1f}%</div>
    <div class="gs-stat-label">CNN Accuracy</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">🚀</div>
    <div class="gs-stat-value" style="color:#22C55E;">{MBV2_ACCURACY:.1f}%</div>
    <div class="gs-stat-label">MobileNetV2 Accuracy</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">📈</div>
    <div class="gs-stat-value" style="color:#4ADE80;">+{improvement:.1f}%</div>
    <div class="gs-stat-label">Accuracy Gain</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">🔋</div>
    <div class="gs-stat-value" style="color:#F87171;">0.63</div>
    <div class="gs-stat-label">Weakest F1 (Plastic, CNN)</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Radar chart ────────────────────────────────────────────────────────
    labels_closed = CLASS_LABELS + [CLASS_LABELS[0]]  # close the polygon
    cnn_vals  = [CNN_F1[c]  for c in CLASS_LABELS] + [CNN_F1[CLASS_LABELS[0]]]
    mbv2_vals = [MBV2_F1[c] for c in CLASS_LABELS] + [MBV2_F1[CLASS_LABELS[0]]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=cnn_vals, theta=labels_closed, fill="toself", name="Custom CNN",
        line=dict(color="#F97316", width=2),
        fillcolor="rgba(249,115,22,.12)",
        hovertemplate="CNN — %{theta}<br>F1: %{r:.2f}<extra></extra>",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=mbv2_vals, theta=labels_closed, fill="toself", name="MobileNetV2",
        line=dict(color="#22C55E", width=2.5),
        fillcolor="rgba(34,197,94,.15)",
        hovertemplate="MobileNetV2 — %{theta}<br>F1: %{r:.2f}<extra></extra>",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#060913",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,.08)",
                            tickfont=dict(color="#64748B", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,.06)", tickfont=dict(color="#94A3B8", size=11)),
        ),
        height=420,
        legend=dict(
            bgcolor="rgba(14,20,36,.7)", font=dict(color="#F1F5F9", size=12),
            bordercolor="rgba(255,255,255,.06)", borderwidth=1,
            x=0.85, y=1.0,
        ),
        title=dict(text="Per-Class F1 Score — Radar Chart", font=dict(size=12, color="#64748B")),
        paper_bgcolor="#060913",
        font=dict(family="Inter"),
        margin=dict(t=60, b=20, l=40, r=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    # ── Per-class bar comparison ───────────────────────────────────────────
    st.markdown('<h2 class="gs-section-title">Per-Class F1 Score</h2>', unsafe_allow_html=True)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Custom CNN", x=CLASS_LABELS,
        y=[CNN_F1[c] for c in CLASS_LABELS],
        marker_color="#F97316", opacity=0.85,
        text=[f"{CNN_F1[c]:.2f}" for c in CLASS_LABELS], textposition="outside",
    ))
    fig_bar.add_trace(go.Bar(
        name="MobileNetV2", x=CLASS_LABELS,
        y=[MBV2_F1[c] for c in CLASS_LABELS],
        marker_color="#22C55E", opacity=0.9,
        text=[f"{MBV2_F1[c]:.2f}" for c in CLASS_LABELS], textposition="outside",
    ))
    fig_bar.update_layout(
        barmode="group", height=320,
        yaxis=dict(range=[0, 1.12], title="F1 Score", gridcolor="rgba(255,255,255,.04)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8")),
        **_LAYOUT,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Accuracy / Precision / Recall bar ──────────────────────────────────
    st.markdown('<h2 class="gs-section-title">Overall Metrics</h2>', unsafe_allow_html=True)

    metrics = ["Accuracy", "Precision", "Recall"]
    cnn_m   = [CNN_ACCURACY, CNN_PRECISION, CNN_RECALL]
    mbv2_m  = [MBV2_ACCURACY, MBV2_PRECISION, MBV2_RECALL]

    fig_ov = go.Figure()
    fig_ov.add_trace(go.Bar(
        name="Custom CNN", x=metrics, y=cnn_m,
        marker_color="#F97316", opacity=0.85,
        text=[f"{v:.1f}%" for v in cnn_m], textposition="outside",
    ))
    fig_ov.add_trace(go.Bar(
        name="MobileNetV2", x=metrics, y=mbv2_m,
        marker_color="#22C55E", opacity=0.9,
        text=[f"{v:.1f}%" for v in mbv2_m], textposition="outside",
    ))
    fig_ov.update_layout(
        barmode="group", height=280,
        yaxis=dict(range=[0, 108], title="Score (%)", gridcolor="rgba(255,255,255,.04)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8")),
        **_LAYOUT,
    )
    st.plotly_chart(fig_ov, use_container_width=True, config={"displayModeBar": False})

    # ── Deployment comparison table ────────────────────────────────────────
    st.markdown('<h2 class="gs-section-title">Deployment Metrics</h2>', unsafe_allow_html=True)

    tflite_size = "~7 MB (INT8)" if (ROOT / "models" / "garbagesort_int8.tflite").exists() else "~7 MB (run export_tflite.py)"
    mbv2_size_mb = round((ROOT / "models" / "transfer_learning_best.h5").stat().st_size / 1e6, 1) \
        if (ROOT / "models" / "transfer_learning_best.h5").exists() else "~28"
    cnn_size_mb = round((ROOT / "models" / "best_model.h5").stat().st_size / 1e6, 1) \
        if (ROOT / "models" / "best_model.h5").exists() else "~222"

    rows = [
        ("Test Accuracy",       f"{CNN_ACCURACY:.2f}%",     f"{MBV2_ACCURACY:.2f}%",    "MobileNetV2 +15.55%"),
        ("Macro F1",            "0.77",                     "0.93",                      "MobileNetV2"),
        ("Model Size (.h5)",    f"{cnn_size_mb} MB",        f"{mbv2_size_mb} MB",        "MobileNetV2 (8×smaller)"),
        ("TFLite INT8 Size",    "~80 MB",                   tflite_size,                 "MobileNetV2"),
        ("Inference (CPU i7)",  "~180 ms/img",              "~95 ms/img",                "MobileNetV2 (~2× faster)"),
        ("Training Epochs",     "50",                       "30 (15+15)",                "MobileNetV2 (faster convergence)"),
        ("Trainable Params",    "~8.5M",                    "~2.3M (top layers only)",   "MobileNetV2 (phase 1)"),
        ("Weakest Class F1",    "Plastic: 0.63",            "Plastic: 0.88",             "MobileNetV2"),
    ]

    rows_html = ""
    for metric, cnn_v, mbv2_v, winner in rows:
        rows_html += f"""
<tr>
  <td style="color:var(--txt2);font-weight:600;">{metric}</td>
  <td style="color:#F97316;">{cnn_v}</td>
  <td style="color:#22C55E;font-weight:700;">{mbv2_v}</td>
  <td style="color:var(--muted);font-size:.8rem;">{winner}</td>
</tr>"""

    st.markdown(f"""
<div class="result-card">
<table class="gs-table">
  <tr>
    <th>Metric</th>
    <th style="color:#F97316;">Custom CNN</th>
    <th style="color:#22C55E;">MobileNetV2</th>
    <th>Winner</th>
  </tr>
  {rows_html}
</table>
</div>
""", unsafe_allow_html=True)

    # ── Decision summary ──────────────────────────────────────────────────
    st.markdown("""
<div class="gs-box eco" style="margin-top:1.5rem;">
  <strong style="color:#4ADE80;">✓ Recommendation: MobileNetV2 for all deployments</strong><br>
  <span style="color:var(--muted);font-size:.88rem;">
    MobileNetV2 outperforms the custom CNN across all 7 classes with +15.55% accuracy,
    while being 8× smaller in H5 format and ~2× faster at inference.
    The TFLite INT8 quantized model shrinks to ~7 MB for Raspberry Pi / Jetson Nano use.
  </span>
</div>
</div>
""", unsafe_allow_html=True)
