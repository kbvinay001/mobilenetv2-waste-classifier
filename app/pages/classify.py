"""classify.py — Single image classification with Grad-CAM + recycling guide."""

import streamlit as st
import numpy as np
from PIL import Image
import io, datetime, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))

from ui_styles import CLASS_COLORS, CLASS_EMOJIS
from components.gradcam import compute_gradcam
from components.qr_utils import get_guide, qr_to_bytes
from components.pdf_report import generate_pdf

CLASS_LABELS = ["Battery", "Cardboard", "Clothes", "Glass", "Metal", "Paper", "Plastic"]
DISPOSAL = {
    "Battery":   "Hazardous waste. Take to a battery recycling centre. Never bin it.",
    "Cardboard": "Flatten and keep dry. Place in the paper/cardboard recycling bin.",
    "Clothes":   "Donate if wearable. Use textile recycling banks for worn-out items.",
    "Glass":     "Rinse clean. Place in a glass-only recycling bin. Remove metal lids first.",
    "Metal":     "Rinse tins/cans. Place in metal or mixed recycling bin.",
    "Paper":     "Keep dry. Recycling bin. Avoid greasy or food-contaminated paper.",
    "Plastic":   "Check resin code (♳–♷). Rinse containers. Most PET/HDPE accepted.",
}


def _preprocess(img: Image.Image) -> np.ndarray:
    img_rgb = img.resize((224, 224)).convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _predict(img: Image.Image, model) -> dict:
    arr = _preprocess(img)
    preds = model.predict(arr, verbose=0)[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    return {
        "class":      CLASS_LABELS[top3_idx[0]],
        "confidence": float(preds[top3_idx[0]] * 100),
        "top3": [{"class": CLASS_LABELS[i], "confidence": float(preds[i] * 100)} for i in top3_idx],
        "pred_index": int(top3_idx[0]),
        "all_probs":  preds,
    }


def show(model, add_log, session_id: str):
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
<div class="gs-hero">
  <div class="gs-welcome-badge"><span class="pulse-dot"></span>AI Classification</div>
  <h1 class="gs-hero-title">
    Classify <span class="text-gradient">Waste Image</span>
  </h1>
  <p class="gs-hero-sub">
    Upload any waste image. MobileNetV2 (93% accuracy) classifies it instantly.
    Toggle Grad-CAM to see exactly <em>where</em> the model looked.
  </p>
</div>
<div class="gs-wrap">
""", unsafe_allow_html=True)

    # ── Upload + Options row ───────────────────────────────────────────────
    col_up, col_opts = st.columns([2, 1], gap="large")
    with col_up:
        uploaded = st.file_uploader(
            "Upload waste image (JPG · PNG · WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="classify_uploader",
        )
    with col_opts:
        st.markdown("""
<div class="gs-box info" style="margin-top:.5rem;">
  <strong style="color:#60A5FA;">How it works</strong><br>
  <span style="color:var(--muted);font-size:.84rem;line-height:2;">
  01 &nbsp;Upload a clear image of waste<br>
  02 &nbsp;Model predicts category + confidence<br>
  03 &nbsp;Grad-CAM shows model focus area<br>
  04 &nbsp;Get recycling guide + QR code<br>
  05 &nbsp;Download annotated image or PDF
  </span>
</div>""", unsafe_allow_html=True)
        use_gradcam = st.toggle("🔍 Grad-CAM Explanation", value=True,
                                help="Adds ~1 s compute. Highly recommended for understanding predictions.")

    if not uploaded:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Load image ────────────────────────────────────────────────────────
    img = Image.open(uploaded).convert("RGB")
    add_log(f"Image uploaded: {uploaded.name} ({uploaded.size/1024:.1f} KB)", "INFO")

    with st.spinner("🤖 Running MobileNetV2 inference…"):
        result = _predict(img, model)

    pred_class = result["class"]
    confidence = result["confidence"]
    top3       = result["top3"]
    color      = CLASS_COLORS.get(pred_class, "#22C55E")
    emoji      = CLASS_EMOJIS.get(pred_class, "♻️")

    add_log(f"Prediction: {pred_class} ({confidence:.1f}%)", "SUCCESS")

    # Store in session for analytics
    if "classification_log" not in st.session_state:
        st.session_state["classification_log"] = []
    st.session_state["classification_log"].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "filename":  uploaded.name,
        "class":     pred_class,
        "confidence": confidence,
    })

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    gradcam_overlay = None
    if use_gradcam:
        with st.spinner("🔍 Computing Grad-CAM…"):
            try:
                gradcam_overlay, _ = compute_gradcam(img, model, result["pred_index"])
                if gradcam_overlay is not None:
                    add_log("Grad-CAM computed successfully.", "SUCCESS")
            except Exception as e:
                add_log(f"Grad-CAM failed: {e}", "WARN")

    # ── Display ───────────────────────────────────────────────────────────
    st.markdown(f"""
<h2 class="gs-section-title" style="margin-top:1.5rem;">Classification Result</h2>
<div class="result-card">
  <span class="class-badge" style="color:{color};border-color:{color};background:{color}18;">
    {emoji} {pred_class}
  </span>
  <span style="margin-left:12px;font-size:.85rem;color:var(--muted);">
    Session: <strong style="color:#fff;">{session_id}</strong>
  </span>
</div>
""", unsafe_allow_html=True)

    # Top-3 confidence bars
    c_left, c_right = st.columns([3, 2], gap="large")
    with c_left:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        for i, item in enumerate(top3):
            bar_color = ["#22C55E", "#FBBF24", "#F87171"][i]
            pct = item["confidence"]
            st.markdown(f"""
<div class="conf-bar-wrap">
  <div class="conf-bar-label">
    <strong>{CLASS_EMOJIS.get(item["class"],"")}&nbsp;{item["class"]}</strong>
    <span style="color:{bar_color};font-weight:700;">{pct:.1f}%</span>
  </div>
  <div class="conf-bar-track">
    <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{bar_color};"></div>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        # All-class probability sparkline table
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.75rem;color:var(--muted);font-weight:700;letter-spacing:.08em;margin-bottom:10px;">ALL CATEGORIES</div>', unsafe_allow_html=True)
        for i, (cls, prob) in enumerate(zip(CLASS_LABELS, result["all_probs"])):
            pct = float(prob * 100)
            c = CLASS_COLORS.get(cls, "#4ADE80")
            bold = "font-weight:800;color:#fff;" if cls == pred_class else ""
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
  <span style="font-size:.75rem;width:72px;{bold}color:var(--txt2);">{CLASS_EMOJIS.get(cls,'')} {cls}</span>
  <div style="flex:1;background:rgba(255,255,255,.06);border-radius:4px;height:5px;">
    <div style="width:{min(pct,100):.1f}%;height:100%;border-radius:4px;background:{c};"></div>
  </div>
  <span style="font-size:.72rem;color:var(--muted);width:36px;text-align:right;">{pct:.1f}%</span>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Images side by side ────────────────────────────────────────────────
    img_cols = [st.columns(2)] if gradcam_overlay is not None else [st.columns(1)]
    if gradcam_overlay is not None:
        col_orig, col_gc = st.columns(2, gap="large")
        with col_orig:
            st.markdown('<div class="result-card"><p style="font-size:.75rem;color:var(--muted);font-weight:700;letter-spacing:.08em;margin-bottom:10px;">ORIGINAL IMAGE</p>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_gc:
            st.markdown('<div class="result-card"><p style="font-size:.75rem;color:var(--muted);font-weight:700;letter-spacing:.08em;margin-bottom:10px;">GRAD-CAM — MODEL ATTENTION</p>', unsafe_allow_html=True)
            st.image(gradcam_overlay, use_container_width=True,
                     caption="Warm areas = where the model focused most")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.image(img, width=300, caption="Uploaded image")

    # ── Recycling guide + QR ──────────────────────────────────────────────
    guide = get_guide(pred_class)
    st.markdown(f"""
<h2 class="gs-section-title" style="margin-top:2rem;">♻ Recycling Guide</h2>
""", unsafe_allow_html=True)

    g_left, g_right = st.columns([3, 1], gap="large")
    with g_left:
        steps_html = "".join(
            f'<li style="color:var(--txt2);margin-bottom:6px;">{s}</li>'
            for s in guide.get("steps", [])
        )
        st.markdown(f"""
<div class="recycle-card">
  <div class="recycle-icon">{guide['emoji']}</div>
  <div class="recycle-title">{guide['title']}</div>
  <p class="recycle-text">{guide['tip']}</p>
  <ol style="margin:12px 0 0 18px;padding:0;">{steps_html}</ol>
</div>""", unsafe_allow_html=True)

    with g_right:
        st.markdown("""
<div class="recycle-card" style="text-align:center;">
  <div class="recycle-title" style="margin-bottom:12px;">SCAN FOR MORE INFO</div>
""", unsafe_allow_html=True)
        try:
            qr_png = qr_to_bytes(guide["url"], size=180)
            st.image(qr_png, width=160)
        except Exception:
            st.markdown(f'<a href="{guide["url"]}" target="_blank" style="color:#4ADE80;">{guide["url"]}</a>', unsafe_allow_html=True)
        st.markdown(f'<a href="{guide["url"]}" target="_blank" style="font-size:.75rem;color:#4ADE80;">Open guide →</a>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Downloads ─────────────────────────────────────────────────────────
    st.markdown('<h2 class="gs-section-title" style="margin-top:2rem;">Download</h2>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3, gap="large")

    # Annotated PNG
    with d1:
        if gradcam_overlay is not None:
            from PIL import Image as PILImg
            buf = io.BytesIO()
            PILImg.fromarray(gradcam_overlay).save(buf, format="PNG")
            st.download_button("⬇ Grad-CAM PNG", buf.getvalue(),
                               f"{pred_class}_gradcam.png", "image/png",
                               use_container_width=True)
        else:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button("⬇ Image PNG", buf.getvalue(),
                               f"{pred_class}.png", "image/png",
                               use_container_width=True)

    # PDF report
    with d2:
        try:
            pdf_bytes = generate_pdf(
                predicted_class=pred_class,
                confidence=confidence,
                top3=top3,
                session_id=session_id,
                gradcam_image=gradcam_overlay,
                original_image=img,
            )
            st.download_button("⬇ PDF Report", pdf_bytes,
                               f"garbagesort_{pred_class.lower()}.pdf", "application/pdf",
                               use_container_width=True)
        except Exception as e:
            st.warning(f"PDF unavailable: {e}")

    # JSON
    with d3:
        import json
        result_json = json.dumps({
            "predicted_class": pred_class,
            "confidence": round(confidence, 2),
            "top3": top3,
            "disposal": DISPOSAL.get(pred_class, ""),
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
        }, indent=2)
        st.download_button("⬇ JSON Result", result_json,
                           f"result_{pred_class.lower()}.json", "application/json",
                           use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
