import requests
import json
import os

# --------------------------------
# API CONFIG
# --------------------------------
API_URL = "http://localhost:8000"

# --------------------------------
# TEST HEALTH ENDPOINT
# --------------------------------
def test_health():
    print("Testing /health endpoint")
    response = requests.get(f"{API_URL}/health")
    print(json.dumps(response.json(), indent=2))
    print()

# --------------------------------
# TEST CLASSES ENDPOINT
# --------------------------------
def test_classes():
    print("Testing /classes endpoint")
    response = requests.get(f"{API_URL}/classes")
    print(json.dumps(response.json(), indent=2))
    print()

# --------------------------------
# TEST SINGLE IMAGE PREDICTION
# --------------------------------
def test_predict(image_path):
    if not os.path.isfile(image_path):
        print("Image not found:", image_path)
        return

    print(f"Testing /predict with image: {image_path}")

    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/predict", files=files)

    result = response.json()

    if result.get("success"):
        print("Prediction successful")
        print("Class      :", result["predicted_class"])
        print("Confidence :", f"{result['confidence']}%")
        print("Disposal   :", result["disposal_instruction"])

        print("\nTop 3 Predictions:")
        for pred in result["top_3_predictions"]:
            print(f"  - {pred['class']}: {pred['confidence']:.2f}%")
    else:
        print("Prediction failed")
        print(result)

    print()

# --------------------------------
# TEST BATCH PREDICTION
# --------------------------------
def test_batch_predict(image_paths):
    print(f"Testing /batch-predict with {len(image_paths)} images")

    files = []
    for img in image_paths:
        if os.path.isfile(img):
            files.append(("files", open(img, "rb")))

    response = requests.post(f"{API_URL}/batch-predict", files=files)

    for _, f in files:
        f.close()

    result = response.json()
    print(json.dumps(result, indent=2))
    print()

# --------------------------------
# MAIN
# --------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("GARBAGESORT AI - API TEST SCRIPT")
    print("=" * 60)
    print()

    test_health()
    test_classes()

    print("Enter image path to test single prediction (or press Enter to skip):")
    img_path = input("> ").strip()

    if img_path:
        test_predict(img_path)
    else:
        print("Single image test skipped")
