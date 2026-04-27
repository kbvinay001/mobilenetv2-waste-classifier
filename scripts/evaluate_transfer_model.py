import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# -----------------------------
# PATHS & SETTINGS
# -----------------------------
MODEL_PATH = "models/transfer_learning_best.h5"
TEST_DIR = r"D:\Vcodez_project\split_dataset\test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading transfer learning model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully\n")

# -----------------------------
# TEST DATA GENERATOR
# -----------------------------
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"Total test images: {test_generator.samples}\n")

# -----------------------------
# EVALUATION
# -----------------------------
print("Evaluating model on test dataset...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print("\n" + "=" * 60)
print("TRANSFER LEARNING MODEL - TEST RESULTS")
print("=" * 60)
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")
print("=" * 60)

# -----------------------------
# PREDICTIONS
# -----------------------------
predictions = model.predict(test_generator, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nCLASSIFICATION REPORT")
print("-" * 60)
report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
)
print(report)

with open("outputs/transfer_learning_classification_report.txt", "w") as f:
    f.write(report)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.title(f"Transfer Learning Confusion Matrix (Accuracy: {test_accuracy * 100:.2f}%)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("outputs/transfer_learning_confusion_matrix.png", dpi=300)
plt.show()

# -----------------------------
# PER-CLASS ACCURACY
# -----------------------------
print("\nPER-CLASS ACCURACY")
print("-" * 60)

class_accuracy = cm.diagonal() / cm.sum(axis=1)

for i, label in enumerate(class_labels):
    correct = cm.diagonal()[i]
    total = cm.sum(axis=1)[i]
    acc = class_accuracy[i] * 100
    print(f"{label:12s}: {correct:3d}/{total:3d} = {acc:6.2f}%")

# -----------------------------
# MODEL COMPARISON
# -----------------------------
ORIGINAL_MODEL_ACC = 77.88

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"Original CNN Model Accuracy      : {ORIGINAL_MODEL_ACC:.2f}%")
print(f"Transfer Learning Model Accuracy : {test_accuracy * 100:.2f}%")
print(f"Improvement                      : +{test_accuracy * 100 - ORIGINAL_MODEL_ACC:.2f}%")
print("=" * 60)

print("\nEvaluation completed successfully.")
print("Results saved in outputs folder.")
