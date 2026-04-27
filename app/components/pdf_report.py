"""pdf_report.py — FPDF2 PDF report generator for GarbageSort AI.

NOTE: Uses only Helvetica (Latin-1). All Unicode/emoji characters have been
replaced with ASCII-safe equivalents to avoid font encoding errors.
"""

import io
import datetime
from pathlib import Path
from fpdf import FPDF
from PIL import Image as PILImage
import numpy as np
import tempfile, os

CLASS_LABELS = ["Battery", "Cardboard", "Clothes", "Glass", "Metal", "Paper", "Plastic"]

DISPOSAL = {
    "Battery":   "Hazardous waste. Take to a designated battery recycling centre. Never throw in general waste.",
    "Cardboard": "Recyclable. Flatten boxes, keep dry, place in paper/cardboard recycling bin.",
    "Clothes":   "Donate if usable. Use textile recycling banks or charity drop-offs. Avoid landfill.",
    "Glass":     "Recyclable. Rinse clean and place in a glass recycling bin. Remove lids.",
    "Metal":     "Recyclable. Clean tins/cans and place in metal recycling. Scrap yards for large items.",
    "Paper":     "Recyclable. Keep dry. Place in paper recycling. Shred confidential documents first.",
    "Plastic":   "Check the resin code (1-7 on base). Most bottles/containers are recyclable after rinsing.",
}

CLASS_ICONS = {
    "Battery": "[BAT]", "Cardboard": "[BOX]", "Clothes": "[CLO]",
    "Glass": "[GLS]", "Metal": "[MTL]", "Paper": "[PAP]", "Plastic": "[PLS]",
}

# Brand colors (RGB)
GREEN  = (34, 197, 94)
DARK   = (6, 9, 19)
CARD   = (14, 20, 36)
WHITE  = (241, 245, 249)
MUTED  = (100, 116, 139)
CYAN   = (34, 211, 238)
RED    = (248, 113, 113)
YELLOW = (251, 191, 36)


def _safe(text: str) -> str:
    """Strip any non-Latin-1 characters so Helvetica never raises a font error."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


class GarbageSortPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_fill_color(*DARK)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*GREEN)
        self.set_xy(8, 4)
        self.cell(0, 10, "GarbageSort AI  |  Smart Waste Classification Report", ln=False)
        self.set_text_color(*MUTED)
        self.set_font("Helvetica", "", 8)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.set_xy(-60, 4)
        self.cell(52, 10, ts, align="R")
        self.ln(14)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 10, f"GarbageSort AI  |  Page {self.page_no()}", align="C")


def _filled_rect(pdf: FPDF, x, y, w, h, r, g, b):
    pdf.set_fill_color(r, g, b)
    pdf.rect(x, y, w, h, "F")


def generate_pdf(
    predicted_class: str,
    confidence: float,
    top3: list[dict],
    session_id: str,
    gradcam_image: np.ndarray | None = None,
    original_image=None,
) -> bytes:
    """
    Generate a single-page PDF classification report.

    Parameters
    ----------
    predicted_class : str
    confidence      : float  (0-100)
    top3            : list of {"class": str, "confidence": float}
    session_id      : str
    gradcam_image   : (224,224,3) uint8 RGB array or None
    original_image  : PIL Image or None

    Returns
    -------
    bytes — PDF binary
    """
    pdf = GarbageSortPDF()
    pdf.add_page()

    # ── Hero band ──────────────────────────────────────────────────────────
    _filled_rect(pdf, 0, 18, 210, 38, *CARD)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(10, 22)
    pdf.cell(0, 10, "Classification Result", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    pdf.set_x(10)
    pdf.cell(0, 8, f"Session ID: {session_id}  |  GarbageSort AI v1.0", ln=True)
    pdf.ln(4)

    # ── Predicted class card ───────────────────────────────────────────────
    y0 = 62
    _filled_rect(pdf, 8, y0, 130, 36, *CARD)
    _filled_rect(pdf, 8, y0, 3, 36, *GREEN)   # green left stripe
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*GREEN)
    pdf.set_xy(16, y0 + 5)
    pdf.cell(0, 6, "PREDICTED CLASS", ln=True)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(16, y0 + 13)
    pdf.cell(0, 12, _safe(predicted_class), ln=True)

    # Confidence card
    conf_x = 145
    _filled_rect(pdf, conf_x, y0, 55, 36, *CARD)
    _filled_rect(pdf, conf_x, y0, 3, 36, *CYAN)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*CYAN)
    pdf.set_xy(conf_x + 8, y0 + 5)
    pdf.cell(0, 6, "CONFIDENCE", ln=True)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(conf_x + 8, y0 + 13)
    pdf.cell(0, 12, f"{confidence:.1f}%", ln=True)

    # ── Top-3 predictions ──────────────────────────────────────────────────
    pdf.set_xy(8, y0 + 44)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, "TOP-3 PREDICTIONS", ln=True)
    pdf.ln(2)

    bar_colors = [GREEN, YELLOW, RED]
    for i, item in enumerate(top3[:3]):
        bx = 10
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*WHITE)
        pdf.set_x(bx)
        pdf.cell(50, 7, _safe(item["class"]), ln=False)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 7, f"{item['confidence']:.1f}%", ln=True, align="R")
        _filled_rect(pdf, bx, pdf.get_y(), 190, 4, 20, 30, 50)
        bar_w = int(190 * item["confidence"] / 100)
        if bar_w > 0:
            _filled_rect(pdf, bx, pdf.get_y(), bar_w, 4, *bar_colors[i])
        pdf.ln(7)

    # ── Images ─────────────────────────────────────────────────────────────
    pdf.ln(4)
    tmp_files = []
    try:
        col_y = pdf.get_y()
        if original_image is not None:
            tmp_orig = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_files.append(tmp_orig.name)
            orig_rgb = original_image.convert("RGB").resize((224, 224))
            orig_rgb.save(tmp_orig.name, "JPEG", quality=90)
            tmp_orig.close()
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*MUTED)
            pdf.set_x(10)
            pdf.cell(95, 6, "ORIGINAL IMAGE", ln=True)
            pdf.image(tmp_orig.name, x=10, y=pdf.get_y(), w=90, h=90)

        if gradcam_image is not None:
            tmp_gc = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_files.append(tmp_gc.name)
            PILImage.fromarray(gradcam_image).save(tmp_gc.name, "JPEG", quality=90)
            tmp_gc.close()
            pdf.set_xy(108, col_y)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*MUTED)
            pdf.cell(95, 6, "GRAD-CAM EXPLANATION", ln=True)
            pdf.image(tmp_gc.name, x=108, y=col_y + 6, w=90, h=90)

        pdf.set_y(col_y + 98)
    finally:
        for f in tmp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

    # ── Disposal instructions ──────────────────────────────────────────────
    pdf.ln(2)
    _filled_rect(pdf, 8, pdf.get_y(), 194, 2, *GREEN)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GREEN)
    pdf.set_x(10)
    pdf.cell(0, 6, "RECYCLING & DISPOSAL INSTRUCTIONS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*WHITE)
    pdf.set_x(10)
    tip = _safe(DISPOSAL.get(predicted_class, "Follow local waste disposal guidelines."))
    pdf.multi_cell(190, 6, tip)

    # ── Footer note ────────────────────────────────────────────────────────
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*MUTED)
    pdf.set_x(10)
    pdf.cell(
        0, 5,
        "Generated by GarbageSort AI  |  MobileNetV2 Transfer Learning Model  |  For informational purposes only.",
        ln=True,
    )

    return bytes(pdf.output())


def generate_batch_pdf(results: list[dict], session_id: str) -> bytes:
    """Generate a summary PDF for batch classification results."""
    pdf = GarbageSortPDF()
    pdf.add_page()

    # Hero
    _filled_rect(pdf, 0, 18, 210, 30, *CARD)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(10, 22)
    pdf.cell(0, 10, "Batch Classification Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.set_x(10)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 7, f"Session: {session_id}  |  {len(results)} images  |  {ts}", ln=True)
    pdf.ln(10)

    # Table header
    col_w = [70, 45, 40, 35]
    headers = ["Filename", "Predicted Class", "Confidence", "Status"]
    pdf.set_fill_color(*CARD)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*MUTED)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 8, h, border=0, fill=True)
    pdf.ln()
    _filled_rect(pdf, 8, pdf.get_y(), 190, 1, *GREEN)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    for r in results:
        name   = _safe(str(r.get("filename", ""))[:30])
        cls    = _safe(str(r.get("class", r.get("predicted_class", "-"))))
        conf   = f"{r.get('confidence', 0):.1f}%"
        is_ok  = r.get("status") == "success"
        status = "OK" if is_ok else "ERR"
        color  = GREEN if is_ok else RED

        pdf.set_text_color(*WHITE)
        pdf.cell(col_w[0], 7, name)
        pdf.cell(col_w[1], 7, cls)
        pdf.cell(col_w[2], 7, conf)
        pdf.set_text_color(*color)
        pdf.cell(col_w[3], 7, status)
        pdf.ln()

    # Summary section
    pdf.ln(6)
    _filled_rect(pdf, 8, pdf.get_y(), 194, 2, *GREEN)
    pdf.ln(4)

    ok_count   = sum(1 for r in results if r.get("status") == "success")
    ok_confs   = [r.get("confidence", 0) for r in results if r.get("status") == "success"]
    avg_conf   = sum(ok_confs) / len(ok_confs) if ok_confs else 0.0

    from collections import Counter
    cls_counts = Counter(r.get("class", "-") for r in results if r.get("status") == "success")
    top_cls    = cls_counts.most_common(1)[0][0] if cls_counts else "-"

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GREEN)
    pdf.set_x(10)
    pdf.cell(0, 6, "BATCH SUMMARY", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*WHITE)
    pdf.set_x(10)
    pdf.cell(0, 6, f"Total: {len(results)}  |  Success: {ok_count}  |  Avg Confidence: {avg_conf:.1f}%  |  Top Class: {_safe(top_cls)}", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*MUTED)
    pdf.set_x(10)
    pdf.cell(0, 5, "Generated by GarbageSort AI  |  MobileNetV2 Transfer Learning  |  For informational purposes only.", ln=True)

    return bytes(pdf.output())
