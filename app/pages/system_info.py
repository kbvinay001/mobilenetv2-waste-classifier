"""system_info.py — Edge device info, model status, and deployment guide."""

import streamlit as st
import platform, datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def _file_mb(path: Path) -> str:
    if path.exists():
        return f"{path.stat().st_size / 1e6:.1f} MB"
    return "Not found"


def _check(path: Path) -> tuple[str, str]:
    if path.exists():
        return "✅", "#4ADE80"
    return "❌", "#F87171"


def show():
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
<div class="gs-hero">
  <div class="gs-welcome-badge"><span class="pulse-dot"></span>System</div>
  <h1 class="gs-hero-title">
    System <span class="text-gradient">Info</span>
  </h1>
  <p class="gs-hero-sub">
    Model file status, edge TFLite deployment guide, system resources,
    and Raspberry Pi setup instructions.
  </p>
</div>
<div class="gs-wrap">
""", unsafe_allow_html=True)

    # ── Model files ────────────────────────────────────────────────────────
    st.markdown('<h2 class="gs-section-title">Model Files</h2>', unsafe_allow_html=True)

    models = [
        ("Custom CNN",              ROOT / "models" / "best_model.h5"),
        ("CNN Final",               ROOT / "models" / "final_model.h5"),
        ("MobileNetV2 Best ★",      ROOT / "models" / "transfer_learning_best.h5"),
        ("MobileNetV2 Final",       ROOT / "models" / "transfer_learning_final.h5"),
        ("TFLite INT8 (Edge)",      ROOT / "models" / "garbagesort_int8.tflite"),
    ]

    rows_html = ""
    for name, path in models:
        icon, col = _check(path)
        size = _file_mb(path)
        is_active = "★" in name
        star_style = "font-weight:800;color:#fff;" if is_active else ""
        rows_html += f"""
<tr>
  <td style="{star_style}color:var(--txt2);">{name}</td>
  <td style="color:var(--muted);font-size:.8rem;font-family:monospace;">{path.name}</td>
  <td style="color:#60A5FA;font-weight:600;">{size}</td>
  <td style="color:{col};">{icon}</td>
</tr>"""

    st.markdown(f"""
<div class="result-card">
<table class="gs-table">
  <tr><th>Model</th><th>Filename</th><th>Size</th><th>Status</th></tr>
  {rows_html}
</table>
</div>""", unsafe_allow_html=True)

    # TFLite export button hint
    tflite_path = ROOT / "models" / "garbagesort_int8.tflite"
    if not tflite_path.exists():
        st.markdown("""
<div class="gs-box warn" style="margin-top:1rem;">
  <strong style="color:#FBBF24;">⚠ TFLite edge model not generated yet</strong><br>
  <span style="color:var(--muted);font-size:.88rem;">
    Run from the project root:<br>
    <code style="background:rgba(255,255,255,.06);padding:2px 8px;border-radius:4px;">
      python edge/export_tflite.py
    </code><br>
    This creates <code>models/garbagesort_int8.tflite</code> (~7 MB, INT8 quantized).
  </span>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="gs-box good">
  <strong style="color:#4ADE80;">✅ TFLite edge model ready</strong>&nbsp;—&nbsp;
  <span style="color:var(--muted);font-size:.88rem;">
    Copy <code>models/garbagesort_int8.tflite</code> to your Raspberry Pi and run
    <code>edge/infer_tflite.py</code>.
  </span>
</div>""", unsafe_allow_html=True)

    # ── System resources ───────────────────────────────────────────────────
    st.markdown('<h2 class="gs-section-title" style="margin-top:2rem;">System Resources</h2>',
                unsafe_allow_html=True)

    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.5)
        mem     = psutil.virtual_memory()
        disk    = psutil.disk_usage(str(ROOT))

        cpu_col  = "#4ADE80" if cpu_pct < 60 else ("#FBBF24" if cpu_pct < 85 else "#F87171")
        mem_col  = "#4ADE80" if mem.percent < 70 else ("#FBBF24" if mem.percent < 88 else "#F87171")

        st.markdown(f"""
<div class="gs-stats-row">
  <div class="gs-stat-card">
    <div class="gs-stat-icon">⚙️</div>
    <div class="gs-stat-value" style="color:{cpu_col};">{cpu_pct:.1f}%</div>
    <div class="gs-stat-label">CPU Usage</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">💾</div>
    <div class="gs-stat-value" style="color:{mem_col};">{mem.percent:.1f}%</div>
    <div class="gs-stat-label">RAM ({mem.used/1e9:.1f} / {mem.total/1e9:.1f} GB)</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">💿</div>
    <div class="gs-stat-value">{disk.percent:.1f}%</div>
    <div class="gs-stat-label">Disk ({disk.used/1e9:.0f} / {disk.total/1e9:.0f} GB)</div>
  </div>
  <div class="gs-stat-card">
    <div class="gs-stat-icon">🖥️</div>
    <div class="gs-stat-value" style="font-size:.9rem;">{platform.processor()[:20] or platform.machine()}</div>
    <div class="gs-stat-label">{platform.system()} {platform.release()}</div>
  </div>
</div>""", unsafe_allow_html=True)
    except ImportError:
        st.info("Install `psutil` to see live system stats: `pip install psutil`")

    # ── Python + TF versions ───────────────────────────────────────────────
    try:
        import tensorflow as tf
        tf_ver = tf.__version__
        tf_gpu = "GPU ✅" if tf.config.list_physical_devices("GPU") else "CPU only"
        gpu_names = ", ".join(
            g.name.split(":")[-1] for g in tf.config.list_physical_devices("GPU")
        ) or "—"
    except Exception:
        tf_ver, tf_gpu, gpu_names = "not found", "—", "—"

    import sys
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    st.markdown(f"""
<div class="result-card">
<table class="gs-table">
  <tr><th>Component</th><th>Value</th></tr>
  <tr><td style="color:var(--txt2);">Python</td><td style="color:#60A5FA;font-weight:600;">{py_ver}</td></tr>
  <tr><td style="color:var(--txt2);">TensorFlow</td><td style="color:#60A5FA;font-weight:600;">{tf_ver}</td></tr>
  <tr><td style="color:var(--txt2);">GPU Support</td><td style="color:#22C55E;font-weight:700;">{tf_gpu}</td></tr>
  <tr><td style="color:var(--txt2);">GPU Device(s)</td><td style="color:#22D3EE;">{gpu_names}</td></tr>
  <tr><td style="color:var(--txt2);">Platform</td><td style="color:var(--muted);">{platform.platform()[:60]}</td></tr>
  <tr><td style="color:var(--txt2);">Report Time</td><td style="color:var(--muted);">{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    # ── Edge deployment guide ─────────────────────────────────────────────
    st.markdown('<h2 class="gs-section-title" style="margin-top:2rem;">🍓 Raspberry Pi 4 Deployment Guide</h2>',
                unsafe_allow_html=True)
    st.markdown("""
<div class="gs-box info">
  <strong style="color:#60A5FA;">Prerequisites: Raspberry Pi 4 (4 GB RAM), Raspberry Pi OS 64-bit, Python 3.10+</strong>
</div>
""", unsafe_allow_html=True)

    steps = [
        ("1. Copy files to Pi",
         "scp models/garbagesort_int8.tflite edge/infer_tflite.py pi@raspberrypi.local:~/garbagesort/"),
        ("2. Install tflite-runtime (lightweight, no full TF needed)",
         "pip install tflite-runtime Pillow numpy"),
        ("3. Run inference on an image",
         "python3 infer_tflite.py --model garbagesort_int8.tflite --image /path/to/waste.jpg"),
        ("4. Expected performance on Pi 4",
         "~250 ms/image (INT8)   vs   ~1500 ms (full TF float32)  →  6× speedup"),
        ("5. For live camera on Pi",
         "python3 infer_tflite.py --model garbagesort_int8.tflite --camera 0 --every 5"),
    ]

    for title, code in steps:
        st.markdown(f"""
<div style="margin-bottom:1rem;">
  <div style="font-size:.85rem;font-weight:600;color:var(--txt2);margin-bottom:6px;">{title}</div>
  <div class="gs-log"><span class="ls">$</span> <span class="li">{code}</span></div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="gs-box eco" style="margin-top:1rem;">
  <strong style="color:#4ADE80;">♻ Why INT8 quantization?</strong><br>
  <span style="color:var(--muted);font-size:.88rem;">
    Post-training INT8 quantization converts 32-bit floats to 8-bit integers.
    Model size drops from ~28 MB → ~7 MB. Inference on Pi 4 improves ~6×.
    Accuracy loss is typically &lt;1% (0.5% observed on this dataset).
  </span>
</div>
</div>
""", unsafe_allow_html=True)
