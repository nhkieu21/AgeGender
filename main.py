from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils import load_models, predict
import io

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

    return {
        "pred_age": pred_age,
        "pred_gender": pred_gender
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
