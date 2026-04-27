"""live_camera.py — Real-time waste classification (webcam + RTSP) for GarbageSort AI."""

import streamlit as st
import cv2
import numpy as np
import threading
import queue
import time
import datetime
from PIL import Image

CLASS_LABELS = ["Battery", "Cardboard", "Clothes", "Glass", "Metal", "Paper", "Plastic"]

CLASS_COLORS_CV = {
    "Battery":   (248, 113, 113),
    "Cardboard": (251, 191, 36),
    "Clothes":   (167, 139, 250),
    "Glass":     (34, 211, 238),
    "Metal":     (96, 165, 250),
    "Paper":     (74, 222, 128),
    "Plastic":   (249, 115, 22),
}

CLASS_EMOJIS = {
    "Battery": "🔋", "Cardboard": "📦", "Clothes": "👕",
    "Glass": "🥛", "Metal": "🥫", "Paper": "📄", "Plastic": "🧴",
}

MAX_RECONNECT = 3


def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, (224, 224)).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def _annotate(frame: np.ndarray, pred_class: str, confidence: float) -> np.ndarray:
    """Draw classification badge on frame."""
    h, w = frame.shape[:2]
    color = CLASS_COLORS_CV.get(pred_class, (34, 197, 94))
    color_bgr = (color[2], color[1], color[0])

    # Overlay box
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 52), (w, h), (6, 9, 19), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Class text
    label = f"{pred_class}  {confidence:.1f}%"
    cv2.putText(frame, label, (12, h - 16),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, color_bgr, 2, cv2.LINE_AA)

    # Accent bar
    bar_w = int(w * confidence / 100)
    cv2.rectangle(frame, (0, h - 4), (bar_w, h), color_bgr, -1)

    # Live indicator
    cv2.circle(frame, (w - 20, 20), 7, (74, 222, 128), -1)
    cv2.putText(frame, "LIVE", (w - 50, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (74, 222, 128), 1, cv2.LINE_AA)

    return frame


def _camera_worker(url, fq: queue.Queue, stop: threading.Event, model, infer_every: int):
    """
    Background thread: read frames → run inference every N frames → push to queue.
    Supports webcam index (int) and RTSP/HTTP URLs (str).
    """
    # Resolve URL
    src = url
    try:
        src = int(url)
    except (ValueError, TypeError):
        src = str(url).strip()

    cap = None
    for attempt in range(1, MAX_RECONNECT + 1):
        fq.put(("LOG", f"Connecting (attempt {attempt}/{MAX_RECONNECT})…"))
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            fq.put(("LOG", f"Camera connected: {src}"))
            break
        cap.release()
        cap = None
        if attempt < MAX_RECONNECT:
            time.sleep(2 ** attempt)

    if cap is None:
        fq.put(("ERR", f"Cannot open camera: {src}"))
        return

    frame_count = 0
    last_pred   = "—"
    last_conf   = 0.0
    fps_t0      = time.time()
    fps_frames  = 0

    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            fq.put(("ERR", "Stream ended or frame dropped."))
            break

        frame_count += 1
        fps_frames  += 1

        # Run inference every N frames (edge-friendly)
        if frame_count % infer_every == 0:
            try:
                arr   = _preprocess(frame)
                preds = model.predict(arr, verbose=0)[0]
                idx   = int(np.argmax(preds))
                last_pred = CLASS_LABELS[idx]
                last_conf = float(preds[idx] * 100)
            except Exception as e:
                fq.put(("LOG", f"Inference error: {e}"))

        # Compute display FPS
        elapsed = time.time() - fps_t0
        display_fps = fps_frames / elapsed if elapsed > 0 else 0.0

        ann = _annotate(frame.copy(), last_pred, last_conf)
        ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

        # Replace oldest item in queue (keep fresh)
        try:
            fq.get_nowait()
        except queue.Empty:
            pass
        fq.put(("FRAME", ann_rgb, last_pred, last_conf, display_fps, frame_count))

    cap.release()
    fq.put(("LOG", "Camera stopped."))


def show(model, add_log, session_id: str):
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
<div class="gs-hero">
  <div class="gs-welcome-badge"><span class="pulse-dot"></span>Live Detection</div>
  <h1 class="gs-hero-title">
    Live <span class="text-gradient">Camera Mode</span>
  </h1>
  <p class="gs-hero-sub">
    Real-time waste classification from webcam or RTSP IP camera.
    Optimised for edge devices — adjustable inference rate keeps Pi 4 responsive.
  </p>
</div>
<div class="gs-wrap">
""", unsafe_allow_html=True)

    # ── Init session state ─────────────────────────────────────────────────
    for k, v in [("live_active", False), ("live_queue", None), ("live_stop", None),
                 ("live_session_log", [])]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Config row ────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 1], gap="large")
    with c1:
        camera_input = st.text_input(
            "Camera source",
            value="0",
            placeholder="0  (webcam)  |  rtsp://user:pass@192.168.1.100:554/stream1  |  http://...",
            help="Enter 0 for built-in webcam, or a full RTSP/HTTP URL for IP cameras.",
        )
    with c2:
        infer_every = st.slider("Infer every N frames", 1, 10, 3,
                                help="Higher = faster display, fewer predictions. Good for Raspberry Pi: 5–10.")

    st.markdown("""
<div class="gs-box info" style="margin-bottom:1.2rem;">
  <strong style="color:#60A5FA;">Supported Sources</strong>&nbsp;&nbsp;
  <span style="color:var(--muted);font-size:.85rem;">
  Webcam → <code>0</code> &nbsp;|&nbsp;
  RTSP IP cam → <code>rtsp://user:pass@192.168.1.x:554/stream1</code> &nbsp;|&nbsp;
  HTTP MJPEG → <code>http://ip:port/video</code>
  </span>
</div>""", unsafe_allow_html=True)

    # ── Start / Stop ──────────────────────────────────────────────────────
    sb1, sb2 = st.columns(2, gap="large")
    start_clicked = sb1.button("▶ Start Camera", type="primary", use_container_width=True)
    stop_clicked  = sb2.button("⏹ Stop Camera", use_container_width=True)

    if start_clicked:
        # Stop any existing stream
        if st.session_state.live_stop:
            st.session_state.live_stop.set()
            time.sleep(0.3)
        fq   = queue.Queue(maxsize=3)
        stop = threading.Event()
        threading.Thread(
            target=_camera_worker,
            args=(camera_input, fq, stop, model, infer_every),
            daemon=True,
        ).start()
        st.session_state.live_queue  = fq
        st.session_state.live_stop   = stop
        st.session_state.live_active = True
        add_log(f"Live stream started: {camera_input}", "INFO")

    if stop_clicked and st.session_state.live_stop:
        st.session_state.live_stop.set()
        st.session_state.live_active = False
        add_log("Camera stopped by user.", "WARN")
        st.rerun()

    # ── Live display loop ─────────────────────────────────────────────────
    if st.session_state.live_active and st.session_state.live_queue:
        st.markdown('<h2 class="gs-section-title" style="margin-top:1.5rem;">Live Feed</h2>', unsafe_allow_html=True)

        col_feed, col_stats = st.columns([3, 1], gap="large")
        with col_feed:
            feed_ph   = st.empty()
            status_ph = st.empty()
        with col_stats:
            st.markdown("<div style='padding-top:.5rem'>", unsafe_allow_html=True)
            cls_ph  = st.empty()
            conf_ph = st.empty()
            fps_ph  = st.empty()
            frm_ph  = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

            # Per-session class distribution
            if st.session_state.live_session_log:
                from collections import Counter
                counts = Counter([r["class"] for r in st.session_state.live_session_log])
                st.markdown("""<div style="margin-top:1rem;font-size:.72rem;color:var(--muted);font-weight:700;letter-spacing:.08em;margin-bottom:8px;">SESSION COUNTS</div>""",
                            unsafe_allow_html=True)
                for cls, cnt in counts.most_common():
                    st.markdown(
                        f'<div style="font-size:.82rem;color:var(--txt2);">'
                        f'{CLASS_EMOJIS.get(cls,"")} {cls}: <strong style="color:#fff;">{cnt}</strong></div>',
                        unsafe_allow_html=True)

        deadline = time.time() + 4.0
        while time.time() < deadline and st.session_state.live_active:
            try:
                msg = st.session_state.live_queue.get(timeout=0.15)
                if msg[0] == "FRAME":
                    _, frame_rgb, pred, conf, fps, frame_no = msg
                    color = {
                        "Battery": "#F87171", "Cardboard": "#FBBF24", "Clothes": "#A78BFA",
                        "Glass": "#22D3EE", "Metal": "#60A5FA", "Paper": "#4ADE80", "Plastic": "#F97316",
                    }.get(pred, "#22C55E")
                    status_ph.markdown(f"""
<div class="gs-hud-status">
  <span class="gs-stream-live">● LIVE</span>
  <span style="font-size:.8rem;font-weight:700;color:{color};">
    {CLASS_EMOJIS.get(pred,"")} {pred}
  </span>
  <span style="font-size:.76rem;color:var(--muted);">{conf:.1f}%</span>
</div>""", unsafe_allow_html=True)
                    feed_ph.image(frame_rgb, use_container_width=True)
                    cls_ph.metric("🗑️ Class",  pred)
                    conf_ph.metric("🎯 Confidence", f"{conf:.1f}%")
                    fps_ph.metric("⚡ FPS",     f"{fps:.1f}")
                    frm_ph.metric("🎞️ Frame",   frame_no)

                    # Log to session
                    st.session_state.live_session_log.append({
                        "class": pred, "confidence": conf,
                        "timestamp": datetime.datetime.now().isoformat(),
                    })
                    # Cap log size
                    if len(st.session_state.live_session_log) > 500:
                        st.session_state.live_session_log = st.session_state.live_session_log[-500:]

                elif msg[0] == "LOG":
                    add_log(msg[1], "INFO")
                elif msg[0] == "ERR":
                    add_log(msg[1], "ALERT")
                    st.session_state.live_active = False
                    status_ph.markdown("""
<div class="gs-hud-status">
  <span class="gs-stream-reconnecting">✗ DISCONNECTED</span>
</div>""", unsafe_allow_html=True)
                    break
            except queue.Empty:
                break
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
