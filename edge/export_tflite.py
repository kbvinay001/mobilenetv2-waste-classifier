"""export_tflite.py — Export MobileNetV2 model to INT8 quantized TFLite for edge deployment."""

import sys
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
import numpy as np
from pathlib import Path

# Add app/ to path so model_utils can be imported
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "app"))

MODEL_PATH   = ROOT / "models" / "transfer_learning_best.h5"
TFLITE_PATH  = ROOT / "models" / "garbagesort_int8.tflite"
DATASET_PATH = ROOT / "split_dataset" / "train"
IMG_SIZE     = (224, 224)
N_CALIBRATION = 100   # images for INT8 calibration


def load_representative_images():
    """Yield N_CALIBRATION random images from the training set for INT8 calibration."""
    image_paths = []
    for class_dir in DATASET_PATH.iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    image_paths.append(img_path)

    rng = np.random.default_rng(42)
    selected = rng.choice(image_paths, size=min(N_CALIBRATION, len(image_paths)), replace=False)
    print(f"Using {len(selected)} calibration images from {DATASET_PATH}")
    return selected


def representative_dataset_gen():
    """Generator for TFLite INT8 calibration."""
    selected = load_representative_images()
    for img_path in selected:
        try:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0).astype(np.float32)
            yield [arr]
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")


def export():
    print("=" * 60)
    print("GarbageSort AI  TFLite INT8 Export")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # [1/4] Load via compatibility loader (handles Keras 2->3 mismatch)
    print(f"\n[1/4] Loading model: {MODEL_PATH.name}")
    from model_utils import load_model_compat
    model, err = load_model_compat(str(MODEL_PATH))
    if err:
        raise RuntimeError(f"Model load failed: {err}")
    print(f"      Input:  {model.input_shape}")
    print(f"      Output: {model.output_shape}")

    # [2/4] Build concrete TF function (required for TFLite converter with tf_keras models)
    print("\n[2/4] Building TFLite converter...")

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)])
    def serving_fn(x):
        return model(x, training=False)

    concrete_fn = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)

    # INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    # [3/4] Calibrate and convert
    print("[3/4] Running INT8 calibration (this may take ~2 min)...")
    tflite_model = converter.convert()

    # [4/4] Save
    TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TFLITE_PATH.write_bytes(tflite_model)

    orig_mb = MODEL_PATH.stat().st_size / 1e6
    lite_mb = TFLITE_PATH.stat().st_size / 1e6
    ratio   = orig_mb / lite_mb

    print(f"\n[4/4] Saved: {TFLITE_PATH}")
    print(f"      Original H5    : {orig_mb:.1f} MB")
    print(f"      TFLite INT8    : {lite_mb:.1f} MB  ({ratio:.1f}x smaller)")
    print("\n  Export complete!")
    print("   Copy models/garbagesort_int8.tflite to your Raspberry Pi.")
    print("   Run: python edge/infer_tflite.py --model models/garbagesort_int8.tflite --image img.jpg")


if __name__ == "__main__":
    export()
