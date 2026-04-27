"""infer_tflite.py — TFLite inference wrapper for GarbageSort AI edge deployment.

Works with only `tflite-runtime` installed — no full TensorFlow needed.
Compatible with Raspberry Pi 4 / Jetson Nano / any Linux ARM device.

Usage
-----
Single image:
    python infer_tflite.py --model garbagesort_int8.tflite --image waste.jpg

Live webcam or RTSP:
    python infer_tflite.py --model garbagesort_int8.tflite --camera 0 --every 5

Install on Pi:
    pip install tflite-runtime Pillow numpy opencv-python-headless
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Force UTF-8 output on Windows (needed for emoji in console)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass  # older Python — ignore, emojis may not render but won't crash


import numpy as _np_check
if tuple(int(x) for x in _np_check.__version__.split(".")[:2]) >= (2, 0):
    import platform
    if platform.machine().startswith(("aarch", "arm")):
        print(
            "\n[WARN] NumPy >= 2.0 detected on ARM. Pre-compiled tflite-runtime wheels\n"
            "       require NumPy < 2.0. If you see an AttributeError, run:\n"
            "         pip3 install 'numpy<2'\n"
            "       Validated fix: AVH (Arm Virtual Hardware) Ubuntu 22.04 Pi 4 session.\n"
        )

CLASS_LABELS = ["Battery", "Cardboard", "Clothes", "Glass", "Metal", "Paper", "Plastic"]
CLASS_EMOJIS = {
    "Battery": "🔋", "Cardboard": "📦", "Clothes": "👕",
    "Glass": "🥛", "Metal": "🥫", "Paper": "📄", "Plastic": "🧴",
}
DISPOSAL = {
    "Battery":   "⚠  Hazardous — battery recycling centre only",
    "Cardboard": "♻  Flatten + paper/cardboard recycling bin",
    "Clothes":   "💚 Donate or textile recycling bank",
    "Glass":     "♻  Rinse + glass recycling bin",
    "Metal":     "♻  Rinse + metal/mixed recycling bin",
    "Paper":     "♻  Keep dry + paper recycling bin",
    "Plastic":   "♻  Check resin code + rinse + recycling bin",
}


def load_interpreter(model_path: str):
    """
    Load TFLite interpreter.
    Priority: ai_edge_litert → tflite_runtime → TensorFlow tf.lite
    """
    # Option 1: ai_edge_litert (TF 2.18+ recommended)
    try:
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        print(f"[ai_edge_litert] Loaded: {model_path}")
        return interp
    except ImportError:
        pass

    # Option 2: tflite-runtime (Raspberry Pi / lightweight)
    try:
        from tflite_runtime.interpreter import Interpreter
        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        print(f"[tflite-runtime] Loaded: {model_path}")
        return interp
    except ImportError:
        pass

    # Option 3: Full TensorFlow (suppress deprecation warning)
    import warnings
    import tensorflow as tf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    print(f"[TensorFlow TFLite] Loaded: {model_path}")
    return interp


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize and normalise to (1, 224, 224, 3) float32."""
    img_rgb = img.convert("RGB").resize((224, 224))
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(interp, arr: np.ndarray) -> tuple[str, float, list]:
    """Run inference; return (class, confidence%, top3 list)."""
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    interp.set_tensor(in_idx, arr)
    t0 = time.perf_counter()
    interp.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000

    preds = interp.get_tensor(out_idx)[0]
    top_idx = int(np.argmax(preds))
    top3 = sorted(
        [{"class": CLASS_LABELS[i], "confidence": round(float(preds[i] * 100), 2)}
         for i in range(len(preds))],
        key=lambda x: x["confidence"], reverse=True,
    )[:3]
    return CLASS_LABELS[top_idx], float(preds[top_idx] * 100), top3, latency_ms


def infer_image(interp, image_path: str):
    """Classify a single image file."""
    img = Image.open(image_path)
    arr = preprocess(img)
    cls, conf, top3, lat = predict(interp, arr)

    print("\n" + "=" * 50)
    print(f"  Image : {Path(image_path).name}")
    print(f"  Class : {CLASS_EMOJIS.get(cls, '')} {cls}")
    print(f"  Conf  : {conf:.2f}%")
    print(f"  Latency: {lat:.1f} ms")
    print("  Top-3 :")
    for i, t in enumerate(top3):
        bar = "█" * int(t["confidence"] / 5)
        print(f"    {i+1}. {t['class']:<12} {t['confidence']:6.2f}%  {bar}")
    print(f"\n  Disposal: {DISPOSAL.get(cls, '')}")
    print("=" * 50)
    return cls, conf


def infer_camera(interp, camera_src, infer_every: int = 5):
    """Real-time classification from webcam or RTSP stream."""
    import cv2

    try:
        src = int(camera_src)
    except ValueError:
        src = camera_src

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera: {src}")
        return

    print(f"\n[LIVE] Camera: {src}  |  Infer every {infer_every} frames")
    print("Press Q to quit.\n")

    frame_no   = 0
    last_cls   = "—"
    last_conf  = 0.0
    last_lat   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        frame_no += 1

        if frame_no % infer_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            arr = preprocess(Image.fromarray(rgb))
            last_cls, last_conf, _, last_lat = predict(interp, arr)
            print(f"Frame {frame_no:5d}  |  {last_cls:<12} {last_conf:6.2f}%  |  {last_lat:.0f} ms")

        # Annotate
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 44), (w, h), (6, 9, 19), -1)
        text = f"{last_cls}  {last_conf:.1f}%  |  {last_lat:.0f} ms/frame"
        cv2.putText(frame, text, (10, h - 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (74, 222, 128), 1, cv2.LINE_AA)
        bar_w = int(w * last_conf / 100)
        cv2.rectangle(frame, (0, h - 4), (bar_w, h), (74, 222, 128), -1)

        cv2.imshow("GarbageSort AI — Edge TFLite", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="GarbageSort AI — TFLite Edge Inference")
    parser.add_argument("--model",  required=True, help="Path to .tflite model file")
    parser.add_argument("--image",  default=None,  help="Image file for single inference")
    parser.add_argument("--camera", default=None,  help="Camera source: 0 (webcam) or RTSP URL")
    parser.add_argument("--every",  type=int, default=5, help="Infer every N frames (camera mode)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        return

    interp = load_interpreter(args.model)

    if args.image:
        infer_image(interp, args.image)
    elif args.camera is not None:
        infer_camera(interp, args.camera, args.every)
    else:
        print("Provide --image or --camera. Example:")
        print("  python infer_tflite.py --model garbagesort_int8.tflite --image waste.jpg")
        print("  python infer_tflite.py --model garbagesort_int8.tflite --camera 0 --every 5")


if __name__ == "__main__":
    main()
