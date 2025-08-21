from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from PIL import Image
import numpy as np
import io
from tempfile import NamedTemporaryFile
import os
from utils import load_models, predict

app = FastAPI(title='Age & Gender Prediction API')

model_age, model_gender = load_models()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to Age and Gender Prediction API"}

@app.post("/predict")
async def predict_from_image(image: UploadFile = File(...)):
    # Kiểm tra định dạng file
    if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse(status_code=400, content={"error": "Only JPG/PNG images are supported"})

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    pred_age, pred_gender = predict(img, model_age, model_gender)

    return {"pred_age": pred_age, "pred_gender": pred_gender}

@app.post("/fas")
async def fas_from_image(image: UploadFile = File(...)):
    if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse(status_code=400, content={"error": "Only JPG/PNG images are supported"})

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        img.save(tmp_path)

    faces = DeepFace.extract_faces(
        img_path=tmp_path,
        enforce_detection=False,
        align=False, 
        anti_spoofing=True
    )

    if faces:
        is_real = faces[0].get("is_real", False)
        results = "REAL" if is_real else "SPOOF"
    else:
        results = None

    return {"face": results}
