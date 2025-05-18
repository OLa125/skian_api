from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoFeatureExtractor, AutoModelForImageClassification
import torch
import io

app = FastAPI()

# Load CLIP model for skin detection
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load skin disease classification model
disease_model = AutoModelForImageClassification.from_pretrained("nateraw/skin-cancer-classifier")
disease_extractor = AutoFeatureExtractor.from_pretrained("nateraw/skin-cancer-classifier")
disease_model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

    # Step 1: Skin detection using CLIP
    texts = ["a photo of human skin", "a photo of something else"]
    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    is_skin = probs[0][0].item() > 0.5

    if not is_skin:
        return JSONResponse(content={"is_skin": False, "diagnosis": None})

    # Step 2: Disease diagnosis
    inputs = disease_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = disease_model(**inputs)
        pred_class = outputs.logits.argmax(-1).item()
        diagnosis = disease_model.config.id2label[pred_class]

    return JSONResponse(content={"is_skin": True, "diagnosis": diagnosis})
