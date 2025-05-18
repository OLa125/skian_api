# Skin Detection and Diagnosis API

This FastAPI application does two things:
1. Uses CLIP model to detect if an image contains human skin.
2. If the image contains skin, it uses a pretrained classifier to detect skin disease.

## Features

- No training required
- Lightweight and fast
- Based on HuggingFace Transformers (CLIP + EfficientNet)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

3. Access Swagger UI:
```
http://localhost:10000/docs
```

## Example Request

```bash
curl -X POST "http://localhost:10000/predict" -F "file=@your_image.jpg"
```