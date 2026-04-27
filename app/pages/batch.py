"""batch.py — Batch image classification page for GarbageSort AI."""

import streamlit as st
import numpy as np
import pandas as pd
import io, zipfile, datetime
from PIL import Image

CLASS_LABELS = ["Battery", "Cardboard", "Clothes", "Glass", "Metal", "Paper", "Plastic"]
DISPOSAL = {
    "Battery":   "Hazardous — battery recycling centre",
    "Cardboard": "Flatten + paper/cardboard bin",
    "Clothes":   "Donate or textile recycling bank",
    "Glass":     "Rinse + glass recycling bin",
    "Metal":     "Rinse + metal recycling bin",
    "Paper":     "Keep dry + paper recycling bin",
    "Plastic":   "Check resin code + rinse + recycling bin",
}
CLASS_EMOJIS = {
    "Battery": "🔋", "Cardboard": "📦", "Clothes": "👕",
    "Glass": "🥛", "Metal": "🥫", "Paper": "📄", "Plastic": "🧴",
}
CLASS_COLORS = {
    "Battery": "#F87171", "Cardboard": "#FBBF24", "Clothes": "#A78BFA",
    "Glass": "#22D3EE", "Metal": "#60A5FA", "Paper": "#4ADE80", "Plastic": "#F97316",
}


def _predict_pil(img: Image.Image, model) -> tuple[str, float, list]:
    img_arr = np.array(img.resize((224, 224)).convert("RGB"), dtype=np.float32) / 255.0
    arr = np.expand_dims(img_arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    top3 = [{"class": CLASS_LABELS[i], "confidence": round(float(preds[i] * 100), 2)}
            for i in np.argsort(preds)[::-1][:3]]
    return CLASS_LABELS[idx], float(preds[idx] * 100), top3


def _collect_images_from_uploads(files) -> list[tuple[str, Image.Image]]:
    """Returns list of (filename, PIL.Image) pairs from uploaded files."""
    result = []
    for f in files:
        if f.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            with zf.open(name) as img_f:
                                result.append((name, Image.open(io.BytesIO(img_f.read())).convert("RGB")))
            except Exception as e:
                st.warning(f"Could not open ZIP {f.name}: {e}")
        else:
            try:
                result.append((f.name, Image.open(io.BytesIO(f.read())).convert("RGB")))
            except Exception:
                pass
    return result[:50]  # Hard limit 50


def show(model, add_log, session_id: str):
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
<div class="gs-hero">
  <div class="gs-welcome-badge"><span class="pulse-dot"></span>Batch Processing</div>
  <h1 class="gs-hero-title">
    Batch <span class="text-gradient">Image Scan</span>
  </h1>
  <p class="gs-hero-sub">
    Upload a ZIP archive or multiple images (up to 50).
    Classify all at once and download results as CSV, JSON, or PDF report.
  </p>
</div>
<div class="gs-wrap">
""", unsafe_allow_html=True)

    # ── Uploader ──────────────────────────────────────────────────────────
    col_up, col_tip = st.columns([2, 1], gap="large")
    with col_up:
        uploaded_files = st.file_uploader(
            "Upload images or a ZIP archive (max 50 images)",
            type=["jpg", "jpeg", "png", "webp", "zip"],
            accept_multiple_files=True,
            key="batch_uploader",
        )
    with col_tip:
        st.markdown("""
<div class="gs-box info" style="margin-top:.5rem;">
  <strong style="color:#60A5FA;">Tips</strong><br>
  <span style="color:var(--muted);font-size:.84rem;line-height:2;">
  • Upload a ZIP to scan an entire folder<br>
  • Images inside nested ZIP subfolders are included<br>
  • Results table sortable by class / confidence<br>
  • CSV + PDF export available after scan
  </span>
</div>""", unsafe_allow_html=True)

    if not uploaded_files:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Collect images ─────────────────────────────────────────────────────
    images = _collect_images_from_uploads(uploaded_files)
    if not images:
        st.warning("No valid images found in the uploaded files.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(f"""
<div class="gs-box good" style="margin-bottom:1rem;">
  <strong style="color:#4ADE80;">✓ {len(images)} images ready</strong>
  <span style="color:var(--muted);font-size:.85rem;"> — click Classify All to begin</span>
</div>""", unsafe_allow_html=True)

    if not st.button("▶ Classify All Images", type="primary", use_container_width=False):
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Run batch inference ────────────────────────────────────────────────
    add_log(f"Batch started: {len(images)} images", "INFO")
    progress_bar = st.progress(0.0)
    status_text  = st.empty()
    results      = []

    for i, (fname, img) in enumerate(images):
        status_text.markdown(
            f'<span style="color:var(--muted);font-size:.85rem;">Processing {i+1}/{len(images)}: {fname}</span>',
            unsafe_allow_html=True,
        )
        try:
            pred_class, confidence, top3 = _predict_pil(img, model)
            results.append({
                "filename":    fname,
                "class":       pred_class,
                "confidence":  round(confidence, 2),
                "disposal":    DISPOSAL.get(pred_class, ""),
                "status":      "success",
            })
        except Exception as e:
            results.append({
                "filename": fname, "class": "—", "confidence": 0.0,
                "disposal": "", "status": f"error: {e}",
            })
        progress_bar.progress((i + 1) / len(images))

    progress_bar.progress(1.0)
    status_text.markdown(
        f'<span style="color:#4ADE80;font-size:.85rem;">✓ Completed {len(results)} images</span>',
        unsafe_allow_html=True,
    )
    add_log(f"Batch done: {len(results)} classified", "SUCCESS")

    # Log to session analytics
    if "classification_log" not in st.session_state:
        st.session_state["classification_log"] = []
    for r in results:
        if r["status"] == "success":
            st.session_state["classification_log"].append({
                "timestamp":  datetime.datetime.now().isoformat(),
                "filename":   r["filename"],
                "class":      r["class"],
                "confidence": r["confidence"],
            })

    # ── Summary stats ──────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    ok = df[df["status"] == "success"]
    avg_conf = ok["confidence"].mean() if len(ok) else 0.0
    top_cls  = ok["class"].mode()[0] if len(ok) else "—"

    st.markdown(f"""
<h2 class="gs-section-title" style="margin-top:2rem;">Results</h2>
<div class="gs-stats-row" style="grid-template-columns:repeat(4,1fr);">
  <div class="gs-stat-card">
    <div class="gs-stat-icon">📦</div>
    <div class="gs-stat-value">{len(results)}</div>
    <div class="gs-stat-label">Images Processed</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">✅</div>
    <div class="gs-stat-value">{len(ok)}</div>
    <div class="gs-stat-label">Successful</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">🎯</div>
    <div class="gs-stat-value">{avg_conf:.1f}%</div>
    <div class="gs-stat-label">Avg Confidence</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">🏆</div>
    <div class="gs-stat-value" style="font-size:1.1rem;">{top_cls}</div>
    <div class="gs-stat-label">Most Common Class</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Plotly mini charts ─────────────────────────────────────────────────
    if len(ok) > 0:
        import plotly.express as px
        import plotly.graph_objects as go

        chart_layout = dict(
            paper_bgcolor="#060913", plot_bgcolor="rgba(14,20,36,.4)",
            font=dict(family="Inter", color="#94A3B8"),
            margin=dict(t=30, b=20, l=20, r=20),
            showlegend=False,
        )

        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            vc = ok["class"].value_counts().reset_index()
            vc.columns = ["class", "count"]
            colors = [CLASS_COLORS.get(c, "#22C55E") for c in vc["class"]]
            fig1 = go.Figure(go.Bar(
                x=vc["class"], y=vc["count"],
                marker_color=colors,
                text=vc["count"], textposition="outside",
            ))
            fig1.update_layout(title="Class Distribution", height=260, **chart_layout)
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        with cc2:
            fig2 = go.Figure(go.Histogram(
                x=ok["confidence"],
                nbinsx=20,
                marker_color="#22C55E",
                opacity=0.8,
            ))
            fig2.update_layout(title="Confidence Distribution", height=260,
                               xaxis_title="Confidence (%)", yaxis_title="Count",
                               **chart_layout)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Results table ──────────────────────────────────────────────────────
    st.markdown('<h2 class="gs-section-title">Classification Table</h2>', unsafe_allow_html=True)

    # Render a styled HTML table
    rows_html = ""
    for r in results:
        c = CLASS_COLORS.get(r["class"], "#4ADE80")
        emoji = CLASS_EMOJIS.get(r["class"], "")
        status_color = "#4ADE80" if r["status"] == "success" else "#F87171"
        rows_html += f"""
<tr>
  <td style="color:var(--txt2);font-size:.8rem;">{r["filename"][:45]}</td>
  <td><span style="color:{c};font-weight:700;">{emoji} {r["class"]}</span></td>
  <td style="color:{c};font-weight:700;">{r["confidence"]:.1f}%</td>
  <td style="color:var(--muted);font-size:.8rem;">{r["disposal"][:50]}</td>
  <td style="color:{status_color};font-size:.8rem;">{"✓" if r["status"]=="success" else "✗"}</td>
</tr>"""

    st.markdown(f"""
<div class="result-card" style="overflow-x:auto;">
<table class="gs-table">
  <tr>
    <th>Filename</th><th>Predicted Class</th><th>Confidence</th>
    <th>Disposal Tip</th><th>Status</th>
  </tr>
  {rows_html}
</table>
</div>""", unsafe_allow_html=True)

    # ── Downloads ──────────────────────────────────────────────────────────
    st.markdown('<h2 class="gs-section-title" style="margin-top:2rem;">Export Results</h2>',
                unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3, gap="large")

    with d1:
        csv = df.to_csv(index=False)
        st.download_button("⬇ CSV Results", csv,
                           f"batch_{session_id}.csv", "text/csv",
                           use_container_width=True)
    with d2:
        import json
        st.download_button("⬇ JSON Results",
                           json.dumps(results, indent=2),
                           f"batch_{session_id}.json", "application/json",
                           use_container_width=True)
    with d3:
        try:
            from components.pdf_report import generate_batch_pdf
            pdf_bytes = generate_batch_pdf(results, session_id)
            st.download_button("⬇ PDF Report", pdf_bytes,
                               f"batch_report_{session_id}.pdf", "application/pdf",
                               use_container_width=True)
        except Exception as e:
            st.warning(f"PDF unavailable: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
