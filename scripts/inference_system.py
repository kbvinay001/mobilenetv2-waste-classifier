import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "models/best_model.h5"
TEST_DATASET_PATH = r"D:\Vcodez_project\split_dataset\test"
IMAGE_SIZE = (224, 224)

CLASS_LABELS = [
    "Battery",
    "Cardboard",
    "Clothes",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully\n")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = np.argmax(preds)
    confidence = preds[pred_idx] * 100

    return CLASS_LABELS[pred_idx], confidence

# -----------------------------
# RUN INFERENCE ON FULL TEST DATASET
# -----------------------------
print("Running inference on FULL test dataset")
print("=" * 60)

total_images = 0
correct_predictions = 0

for true_class in os.listdir(TEST_DATASET_PATH):
    class_path = os.path.join(TEST_DATASET_PATH, true_class)

    if not os.path.isdir(class_path):
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\nFolder: {true_class} | Images: {len(images)}")
    print("-" * 50)

    for img_name in images:
        img_path = os.path.join(class_path, img_name)

        predicted_class, confidence = predict_image(img_path)

        print(
            f"{img_name:30s} | "
            f"Predicted: {predicted_class:10s} | "
            f"Confidence: {confidence:6.2f}%"
        )

        total_images += 1
        if predicted_class == true_class:
            correct_predictions += 1

# -----------------------------
# FINAL SUMMARY
# -----------------------------
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0

print("\n" + "=" * 60)
print("INFERENCE SUMMARY")
print("=" * 60)
print(f"Total images tested : {total_images}")
print(f"Correct predictions : {correct_predictions}")
print(f"Inference accuracy  : {accuracy:.2f}%")
print("\nInference completed successfully.")
