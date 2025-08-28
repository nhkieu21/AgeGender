from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from utils import load_models, predict, fas_face

app = FastAPI(title='Age & Gender Prediction API')

model_age, model_gender, fas_model = load_models()

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
async def fas(image: UploadFile = File(...)):
    if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse(status_code=400, content={"error":"Only JPG/PNG images are supported"})
    
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error":"Invalid image file"})
        
    result = fas_face(img, fas_model)
    is_real = "REAL" if result == 0 else "SPOOF"

    return {"face" : is_real}
