"""qr_utils.py — QR code generation + per-class recycling guides for GarbageSort AI."""

import io
import qrcode
from PIL import Image

# ── Per-class recycling guides ─────────────────────────────────────────────────
RECYCLING_GUIDE = {
    "Battery": {
        "emoji":    "🔋",
        "title":    "Hazardous — Battery Recycling",
        "tip":      (
            "Batteries contain toxic heavy metals (lead, cadmium, mercury). "
            "NEVER throw in general waste or recycling bins. "
            "Take to a dedicated battery drop-off point — most supermarkets, "
            "electronics shops, and councils offer free collection."
        ),
        "steps": [
            "Tape the terminals with clear tape to prevent short circuits.",
            "Store in a cool, dry place until drop-off.",
            "Find your nearest drop-off at the link below.",
        ],
        "url":      "https://www.batteryback.org/find-a-site/",
        "color":    "#F87171",
    },
    "Cardboard": {
        "emoji":    "📦",
        "title":    "Recyclable — Cardboard",
        "tip":      (
            "Cardboard is one of the most recyclable materials. "
            "Flatten all boxes to save space. Keep dry — wet cardboard "
            "cannot be processed and should go in general waste."
        ),
        "steps": [
            "Remove all tape, staples, and polystyrene inserts.",
            "Flatten boxes completely.",
            "Place in the cardboard/paper recycling bin.",
        ],
        "url":      "https://www.recyclenow.com/what-to-do-with/cardboard",
        "color":    "#FBBF24",
    },
    "Clothes": {
        "emoji":    "👕",
        "title":    "Donate or Textile Recycle — Clothes",
        "tip":      (
            "Textiles can almost always be reused or recycled. "
            "If still wearable, donate to charity shops or clothing banks. "
            "Worn-out items go to textile recycling — not landfill."
        ),
        "steps": [
            "Bag wearable items separately from worn-out ones.",
            "Drop wearable clothes at a charity shop or clothing bank.",
            "Place worn textiles in a textile recycling bank.",
        ],
        "url":      "https://www.recyclenow.com/what-to-do-with/clothes-and-textiles",
        "color":    "#A78BFA",
    },
    "Glass": {
        "emoji":    "🥛",
        "title":    "Recyclable — Glass",
        "tip":      (
            "Glass is infinitely recyclable without quality loss. "
            "Rinse containers before recycling. Not all glass is the same — "
            "window glass and Pyrex should NOT go in bottle banks."
        ),
        "steps": [
            "Rinse bottles and jars — labels can be left on.",
            "Remove metal lids (recycle separately as metal).",
            "Place in a glass-only recycling bin or bottle bank.",
        ],
        "url":      "https://www.recyclenow.com/what-to-do-with/glass-0",
        "color":    "#22D3EE",
    },
    "Metal": {
        "emoji":    "🥫",
        "title":    "Recyclable — Metal",
        "tip":      (
            "Steel and aluminium are valuable recyclables. "
            "Clean tins and cans go in your kerbside bin. "
            "Larger metal items should go to a scrap metal dealer or household waste site."
        ),
        "steps": [
            "Rinse out food and drink cans.",
            "Squash cans to save space if permitted.",
            "Place in the metal/mixed recycling bin.",
        ],
        "url":      "https://www.recyclenow.com/what-to-do-with/metal-tins-cans",
        "color":    "#60A5FA",
    },
    "Paper": {
        "emoji":    "📄",
        "title":    "Recyclable — Paper",
        "tip":      (
            "Paper is highly recyclable but must be kept dry and clean. "
            "Greasy or food-contaminated paper (e.g., pizza boxes) "
            "should NOT go in paper recycling."
        ),
        "steps": [
            "Keep paper dry — wet paper is not recyclable.",
            "Shred confidential documents, then recycle the shreds.",
            "Place newspapers, magazines, and office paper in the paper bin.",
        ],
        "url":      "https://www.recyclenow.com/what-to-do-with/paper-0",
        "color":    "#4ADE80",
    },
    "Plastic": {
        "emoji":    "🧴",
        "title":    "Check & Recycle — Plastic",
        "tip":      (
            "Not all plastics are equal. Check the resin identification code (♳–♷) "
            "on the base. Types 1 (PET) and 2 (HDPE) are widely accepted. "
            "Types 3, 6, and 7 are often not accepted kerbside."
        ),
        "steps": [
            "Check the resin code on the bottom of the item.",
            "Rinse containers to remove food residue.",
            "Place accepted plastics in the recycling bin; others in general waste.",
        ],
        "url":      "https://www.recyclenow.com/what-to-do-with/plastic-bottles",
        "color":    "#F97316",
    },
}


def get_guide(class_name: str) -> dict:
    """Return the recycling guide dict for a given class."""
    return RECYCLING_GUIDE.get(class_name, {
        "emoji": "♻️",
        "title": "General Recycling",
        "tip": "Follow your local council guidelines for proper disposal.",
        "steps": ["Check local council website for guidance."],
        "url": "https://www.recyclenow.com/",
        "color": "#22C55E",
    })


def make_qr_image(url: str, size: int = 200) -> Image.Image:
    """
    Generate a QR code PIL image for a recycling guide URL.

    Parameters
    ----------
    url  : str
    size : int  — output pixel size

    Returns
    -------
    PIL Image (RGBA)
    """
    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=8,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(
        fill_color="#22C55E",
        back_color="#060913",
    ).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def qr_to_bytes(url: str, size: int = 200) -> bytes:
    """Return QR code as PNG bytes."""
    img = make_qr_image(url, size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
