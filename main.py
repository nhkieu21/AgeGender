from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils import load_models, predict
from deepface import DeepFace
import io
import os
import numpy as np

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

# async def predict_from_image(image: UploadFile = File(...)):
#     # Kiểm tra định dạng file
#     if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
#         return JSONResponse(status_code=400, content={"error": "Only JPG/PNG images are supported"})

#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception:
#         return JSONResponse(status_code=400, content={"error": "Invalid image file"})

#     pred_age, pred_gender = predict(img, model_age, model_gender)

#     return {
#         "pred_age": pred_age,
#         "pred_gender": pred_gender
#     }

async def predict_from_image(image: UploadFile = File(...)):
    # Kiểm tra định dạng file
    if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse(status_code=400, content={"error": "Only JPG/PNG images are supported"})

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)   # chuyển sang numpy
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    try:
        faces = DeepFace.extract_faces(
            img_path=img_np,       # truyền thẳng numpy array
            enforce_detection=False,
            anti_spoofing=True
        )
        results = []
        for face in faces:
            is_real = face.get("is_real", False)
            results.append({
                "is_real": bool(is_real),
                "result": "REAL" if is_real else "SPOOF"
            })

        return {"faces": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))  # Lấy PORT từ Render
#     uvicorn.run(app, host="0.0.0.0", port=port)
