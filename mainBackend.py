# mainBackend.py
# 404-Found AI Backend — Fixed Version

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import shutil
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# DEEPFAKE MODEL IMPORTS
# -----------------------------
import torch
import timm
import numpy as np
import pickle
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------
# FAKE NEWS IMPORTS
# -----------------------------
from groq import Groq
from dotenv import load_dotenv

# -----------------------------
# OCR IMPORT
# -----------------------------
import easyocr

load_dotenv()

# ==============================
# APP CONFIG
# ==============================

app = FastAPI(
    title="404-Found AI Backend",
    description="Fake News + Deepfake Detection API",
    version="2.0.0"
)

# ==============================
# CORS
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# LOAD DEEPFAKE MODEL AT STARTUP
# ==============================

logger.info("Loading deepfake detection model...")

# ResNet18 as feature extractor (num_classes=0 removes the classifier head)
resnet = timm.create_model('resnet18', pretrained=True, num_classes=0)
resnet.eval()

# ImageNet normalization — required for correct ResNet features
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the trained Logistic Regression classifier
if not os.path.exists("fake_detector.pkl"):
    raise FileNotFoundError(
        "fake_detector.pkl not found. "
        "Please run train_model.py (sklearn version) first to generate it."
    )

with open("fake_detector.pkl", "rb") as f:
    clf = pickle.load(f)

logger.info("Deepfake model loaded successfully!")

# ==============================
# LOAD GROQ CLIENT FOR FAKE NEWS
# ==============================

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise EnvironmentError(
        "GROQ_API_KEY not found in environment. "
        "Please add it to your .env file."
    )

groq_client = Groq(api_key=groq_api_key)
logger.info("Groq client initialized!")

# ==============================
# LOAD OCR READER
# ==============================

logger.info("Loading EasyOCR reader (first run may take a moment)...")
ocr_reader = easyocr.Reader(['en'], gpu=False)
logger.info("OCR reader ready!")

# ==============================
# UPLOAD FOLDER
# ==============================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# REQUEST MODELS
# ==============================

class NewsRequest(BaseModel):
    text: str

# ==============================
# HELPER: EXTRACT IMAGE FEATURES
# ==============================

def extract_features(image_path: str) -> np.ndarray:
    """Extract ResNet18 features from an image with proper normalization."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = image_transform(img).unsqueeze(0)  # shape: (1, 3, 224, 224)

    with torch.no_grad():
        features = resnet(img_tensor)  # shape: (1, 512)

    return features.numpy().flatten()

# ==============================
# HELPER: ANALYZE NEWS WITH GROQ
# ==============================

def analyze_news_with_groq(text: str) -> dict:
    """Call LLaMA via Groq to analyze if news is real or fake."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this news and tell me if it is REAL or FAKE.

News: {text}

Reply in this exact format only:
VERDICT: REAL or FAKE
SCORE: (0-100, where 100 = definitely real)
REASON: (one short sentence)"""
            }
        ]
    )

    raw = response.choices[0].message.content.strip()

    result = {
        "verdict": "UNKNOWN",
        "credibility_score": 50,
        "reason": "Could not analyze"
    }

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("VERDICT:"):
            result["verdict"] = line.replace("VERDICT:", "").strip()
        elif line.startswith("SCORE:"):
            try:
                result["credibility_score"] = int(line.replace("SCORE:", "").strip())
            except ValueError:
                result["credibility_score"] = 50
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()

    return result

# ==============================
# HOME ROUTE
# ==============================

@app.get("/")
def home():
    return {"message": "404-Found Backend Running Successfully"}

# ==============================
# FAKE NEWS DETECTION
# ==============================

@app.post("/predict-news")
def predict_news(request: NewsRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(text) < 20:
        raise HTTPException(
            status_code=400,
            detail="Text is too short to analyze. Please provide more content."
        )

    try:
        result = analyze_news_with_groq(text)
    except Exception as e:
        logger.error(f"News analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"News analysis failed: {str(e)}"
        )

    return {
        "prediction": result["verdict"],
        "credibility_score": result["credibility_score"],
        "reason": result["reason"]
    }

# ==============================
# DEEPFAKE DETECTION
# ==============================

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

@app.post("/detect-deepfake")
async def detect_deepfake(file: UploadFile = File(...)):

    # Validate file extension
    filename = file.filename or ""
    if "." not in filename:
        raise HTTPException(status_code=400, detail="File has no extension")

    file_extension = filename.rsplit(".", 1)[-1].lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_extension}'. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract features using ResNet18
        features = extract_features(file_path)

        # Predict using trained Logistic Regression
        prediction_label = clf.predict([features])[0]

        # Get probability/confidence
        proba = clf.predict_proba([features])[0]
        confidence = round(float(max(proba)) * 100, 2)

        # FIX: backend was returning "Fake Image"/"Real Image"
        # but frontend was checking for "Deepfake"/"Real Image"
        # Standardized to match frontend expectations
        prediction = "Deepfake" if prediction_label == 1 else "Real Image"

    except Exception as e:
        logger.error(f"Deepfake detection error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Deepfake detection failed: {str(e)}"
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": f"{confidence}%"
    }

# ==============================
# OCR ENDPOINT (NEW)
# ==============================

@app.post("/extract-text")
async def extract_text_from_image(file: UploadFile = File(...)):
    """Extract text from an uploaded image using EasyOCR."""

    filename = file.filename or ""
    if "." not in filename:
        raise HTTPException(status_code=400, detail="File has no extension")

    file_extension = filename.rsplit(".", 1)[-1].lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_extension}'. Allowed: jpg, jpeg, png"
        )

    unique_filename = f"ocr_{uuid.uuid4().hex}.{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ocr_reader.readtext(file_path, detail=0)
        extracted_text = " ".join(result).strip()

        if not extracted_text:
            extracted_text = "No text found in image"

    except Exception as e:
        logger.error(f"OCR error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {str(e)}"
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return {
        "filename": file.filename,
        "extracted_text": extracted_text,
        "word_count": len(extracted_text.split()) if extracted_text else 0
    }

# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models": {
            "deepfake_detector": "loaded",
            "ocr_reader": "loaded",
            "groq_client": "initialized"
        }
    }

# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":
    uvicorn.run(
        "mainBackend:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
