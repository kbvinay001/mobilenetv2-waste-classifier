"""model_utils.py — Architecture rebuild + weight loading for Keras 2/3 compatibility.

When tf.keras.models.load_model fails due to Keras version mismatch
(e.g., 'batch_shape' deserialization error), we rebuild the exact
MobileNetV2 Sequential architecture and load weights from the H5 file.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np

# ── Architecture constants (must match transfer_learning_model.py exactly) ─────
NUM_CLASSES  = 7
IMG_SIZE     = (224, 224, 3)


def build_mobilenetv2_model(num_classes: int = NUM_CLASSES):
    """
    Rebuild the MobileNetV2 Sequential model from scratch.
    Matches the architecture in scripts/transfer_learning_model.py exactly.
    """
    import tf_keras
    from tf_keras.applications import MobileNetV2
    from tf_keras.models import Sequential
    from tf_keras.layers import Dense, Dropout, GlobalAveragePooling2D

    base = MobileNetV2(
        weights=None,            # No ImageNet weights — we'll load from H5
        include_top=False,
        input_shape=IMG_SIZE,
    )
    base.trainable = True
    for layer in base.layers[:100]:
        layer.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_weights_from_h5(model, h5_path: str):
    """
    Load weights layer-by-layer from an H5 file saved with model.save().
    Works even when the architecture deserialization fails.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        # Keras H5 files store weights under 'model_weights' group
        # Try standard H5 weight loading
        try:
            model.load_weights(h5_path, by_name=True, skip_mismatch=False)
            return True, None
        except Exception as e1:
            pass

        # Fallback: manual layer-by-layer extraction
        try:
            weight_group = f.get("model_weights") or f
            for layer in model.layers:
                if layer.name in weight_group:
                    g = weight_group[layer.name]
                    layer_weights = []
                    # Keras stores weight names as attrs
                    weight_names = [n.decode() if isinstance(n, bytes) else n
                                    for n in g.attrs.get("weight_names", [])]
                    for wn in weight_names:
                        layer_weights.append(np.array(g[wn]))
                    if layer_weights:
                        layer.set_weights(layer_weights)
            return True, None
        except Exception as e2:
            return False, f"by_name: {e1} | manual: {e2}"


def load_model_compat(h5_path: str):
    """
    Full compatibility loader:
      1. Try tf_keras.models.load_model (fast path)
      2. Rebuild architecture + load weights (guaranteed compatible path)

    Returns (model, error_str_or_None)
    """
    import tf_keras

    # ── Fast path: standard load ───────────────────────────────────────────
    try:
        model = tf_keras.models.load_model(h5_path, compile=False)
        _ = model.predict(np.zeros((1, *IMG_SIZE), dtype=np.float32), verbose=0)
        return model, None
    except Exception as fast_err:
        pass

    # ── Rebuild + load weights ─────────────────────────────────────────────
    try:
        model = build_mobilenetv2_model(NUM_CLASSES)
        ok, err = load_weights_from_h5(model, h5_path)
        if not ok:
            return None, f"Rebuild failed: {err}"
        # Validate
        out = model.predict(np.zeros((1, *IMG_SIZE), dtype=np.float32), verbose=0)
        if out.shape[-1] != NUM_CLASSES:
            return None, f"Bad output shape: {out.shape}"
        return model, None
    except Exception as rebuild_err:
        return None, f"Fast: {fast_err} | Rebuild: {rebuild_err}"
