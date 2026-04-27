"""gradcam.py — Grad-CAM visual explanation for MobileNetV2 Sequential model."""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import cv2
import tf_keras
import tensorflow as tf
from PIL import Image


def _find_last_conv(mobilenet_model):
    """Walk MobileNetV2 layers in reverse to find the last spatial (4-D) layer."""
    candidates = ["out_relu", "Conv_1_bn", "Conv_1", "block_16_project_BN",
                  "block_16_project"]
    for name in candidates:
        try:
            lyr = mobilenet_model.get_layer(name)
            if len(lyr.output.shape) == 4:
                return name
        except Exception:
            continue
    # Fallback: scan reversed layers
    for lyr in reversed(mobilenet_model.layers):
        try:
            if len(lyr.output.shape) == 4:
                return lyr.name
        except Exception:
            continue
    return None


def compute_gradcam(img_pil: Image.Image, model, pred_index: int | None = None):
    """
    Compute Grad-CAM heatmap for a PIL image using the Sequential MobileNetV2 model.

    Returns
    -------
    heatmap_overlay : np.ndarray  (224×224×3, uint8) — image with CAM overlay
    heatmap_raw     : np.ndarray  (H×W, float32)     — raw [0,1] heatmap
    """
    # ── Build gradient model ───────────────────────────────────────────────
    mobilenet = model.layers[0]       # MobileNetV2 functional sub-model
    conv_name = _find_last_conv(mobilenet)
    if conv_name is None:
        return None, None

    conv_layer = mobilenet.get_layer(conv_name)

    # One Model: input → [last_conv_output, final_prediction]
    grad_model = tf_keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )

    # ── Preprocess ────────────────────────────────────────────────────────
    img_resized = img_pil.resize((224, 224)).convert("RGB")
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)
    img_tensor = tf.cast(img_array, tf.float32)

    # ── Forward + gradient pass ───────────────────────────────────────────
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_output, preds = grad_model(img_tensor)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, conv_output)   # ∂score/∂conv
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # global avg pooling

    # Weighted combination
    conv_out_np = conv_output.numpy()[0]               # (H, W, C)
    pooled_np   = pooled_grads.numpy()                 # (C,)
    heatmap     = conv_out_np @ pooled_np              # (H, W)

    # ReLU + normalise
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # ── Overlay on original image ─────────────────────────────────────────
    img_np = np.array(img_resized, dtype=np.uint8)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend: 60% original + 40% heatmap
    overlay = cv2.addWeighted(img_np, 0.60, heatmap_rgb, 0.40, 0)

    return overlay, heatmap
