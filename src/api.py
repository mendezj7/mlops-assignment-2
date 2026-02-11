import io
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic response models
class HealthResponse(BaseModel):
    status: str


class PredictionResponse(BaseModel):
    predicted_label: str
    probabilities: dict
    confidence: float


class ErrorResponse(BaseModel):
    error: str


class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu")
    class_to_idx = ckpt.get("class_to_idx", {"Cat": 0, "Dog": 1})
    model = SimpleCNN(num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    img_size = ckpt.get("img_size", 128)
    return model, class_to_idx, img_size


def preprocess_image(img: Image.Image, img_size: int) -> torch.Tensor:
    img = img.convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    x = torch.from_numpy(arr).unsqueeze(0)
    return x


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "artifacts/models/baseline_cnn.pt"))
if not MODEL_PATH.exists():
    logger.error(f"Model not found at {MODEL_PATH}. Run training or dvc pull.")
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training or dvc pull.")

try:
    model, class_to_idx, IMG_SIZE = load_model(MODEL_PATH)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"Classes: {list(class_to_idx.keys())}, Image size: {IMG_SIZE}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

app = FastAPI(title="Cats vs Dogs Classifier", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint to verify the service is running."""
    logger.info("Health check request received")
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image.
    
    Args:
        file: Image file (jpg, png, jpeg, bmp)
    
    Returns:
        PredictionResponse with predicted label and probabilities
    
    Raises:
        HTTPException: If image is invalid or processing fails
    """
    logger.info(f"Prediction request received for file: {file.filename}")
    
    # Validate file size (max 10MB)
    max_size = 10 * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        logger.warning(f"File too large: {len(content)} bytes > {max_size} bytes")
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {max_size / (1024*1024):.1f}MB"
        )
    
    # Validate file type
    allowed_formats = {".jpg", ".jpeg", ".png", ".bmp"}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in allowed_formats:
        logger.warning(f"Invalid file format: {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(allowed_formats)}"
        )
    
    try:
        # Load and validate image
        img = Image.open(io.BytesIO(content))
        logger.info(f"Image loaded successfully. Size: {img.size}, Format: {img.format}")
        
        # Preprocess
        x = preprocess_image(img, IMG_SIZE)
        logger.debug(f"Image preprocessed. Tensor shape: {x.shape}")
        
        # Predict
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        pred_idx = int(np.argmax(probs))
        pred_label = idx_to_class[pred_idx]
        confidence = float(probs[pred_idx])
        
        logger.info(f"Prediction: {pred_label} (confidence: {confidence:.4f})")
        
        return PredictionResponse(
            predicted_label=pred_label,
            probabilities={idx_to_class[i]: float(p) for i, p in enumerate(probs)},
            confidence=confidence
        )
    
    except Image.UnidentifiedImageError:
        logger.error("Failed to identify image format")
        raise HTTPException(
            status_code=400,
            detail="Failed to identify image format. Ensure file is a valid image."
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
