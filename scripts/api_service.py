from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import uvicorn
from datetime import datetime
import os

# -------------------------------------------------
# APP INITIALIZATION
# -------------------------------------------------
app = FastAPI(
    title="GarbageSort AI API",
    description="Smart Waste Classification using Transfer Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "models/transfer_learning_best.h5"
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

DISPOSAL_INSTRUCTIONS = {
    "Battery": "Hazardous waste. Take to a battery recycling center.",
    "Cardboard": "Recyclable. Flatten and place in recycling bin.",
    "Clothes": "Donate if usable or use textile recycling.",
    "Glass": "Recyclable. Rinse and place in glass recycling bin.",
    "Metal": "Recyclable. Clean and place in metal recycling bin.",
    "Paper": "Recyclable. Keep dry and place in paper recycling bin.",
    "Plastic": "Check plastic type. Most are recyclable after rinsing."
}

model = None

# -------------------------------------------------
# STARTUP EVENT
# -------------------------------------------------
@app.on_event("startup")
async def load_model_on_startup():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found")

    model = load_model(MODEL_PATH)
    print("Model loaded successfully")

# -------------------------------------------------
# BASIC ENDPOINTS
# -------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Welcome to GarbageSort AI API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/classes")
async def get_classes():
    return {
        "classes": CLASS_LABELS,
        "total_classes": len(CLASS_LABELS)
    }

# -------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------
def preprocess_image(img: Image.Image):
    img = img.resize(IMAGE_SIZE)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------------------------
# SINGLE IMAGE PREDICTION
# -------------------------------------------------
@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        processed = preprocess_image(img)

        preds = model.predict(processed, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx] * 100)

        top_3_idx = np.argsort(preds)[-3:][::-1]
        top_3 = [
            {
                "class": CLASS_LABELS[i],
                "confidence": round(float(preds[i] * 100), 2)
            }
            for i in top_3_idx
        ]

        return {
            "success": True,
            "predicted_class": CLASS_LABELS[idx],
            "confidence": round(confidence, 2),
            "top_3_predictions": top_3,
            "disposal_instruction": DISPOSAL_INSTRUCTIONS[CLASS_LABELS[idx]],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# BATCH PREDICTION
# -------------------------------------------------
@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed")

    results = []

    for file in files:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
            processed = preprocess_image(img)

            preds = model.predict(processed, verbose=0)[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx] * 100)

            results.append({
                "filename": file.filename,
                "predicted_class": CLASS_LABELS[idx],
                "confidence": round(confidence, 2),
                "status": "success"
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })

    return {
        "success": True,
        "total_images": len(files),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

# -------------------------------------------------
# RUN SERVER
# -------------------------------------------------
if __name__ == "__main__":
    print("Starting GarbageSort AI API")
    print("API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
