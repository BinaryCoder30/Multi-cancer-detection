from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import os

app = FastAPI(title="Multi-Cancer Prediction API")

# Define paths
ORGAN_MODEL_PATH = "/Users/krishdhamecha/Downloads/multi_cancer_webapp/organ_model.pth"
COLON_MODEL_PATH = "/Users/krishdhamecha/Downloads/multi_cancer_webapp/colon_model.pth"
LUNG_MODEL_PATH = "/Users/krishdhamecha/Downloads/multi_cancer_webapp/lung_model.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Helper to create model architecture ---
def create_model(num_classes, hidden_units=256):
    model = models.efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, num_classes)
    )
    return model

# --- Load Models ---
model_organ = create_model(num_classes=2, hidden_units=128)
try:
    model_organ.load_state_dict(torch.load(ORGAN_MODEL_PATH, map_location=device))
    model_organ.to(device)
    model_organ.eval()
    print("Organ model loaded successfully!")
except Exception as e:
    print(f"Error loading Organ model: {e}")

model_colon = create_model(num_classes=2, hidden_units=256)
try:
    model_colon.load_state_dict(torch.load(COLON_MODEL_PATH, map_location=device))
    model_colon.to(device)
    model_colon.eval()
    print("Colon model loaded successfully!")
except Exception as e:
    print(f"Error loading Colon model: {e}")

model_lung = create_model(num_classes=3, hidden_units=256)
try:
    model_lung.load_state_dict(torch.load(LUNG_MODEL_PATH, map_location=device))
    model_lung.to(device)
    model_lung.eval()
    print("Lung model loaded successfully!")
except Exception as e:
    print(f"Error loading Lung model: {e}")

ORGAN_CLASSES = {0: "lung", 1: "colon"}
CLASS_NAMES_COLON = ["Colon Adenocarcinoma", "Colon Normal"]
CLASS_NAMES_LUNG = ["Lung Adenocarcinoma", "Lung Benign Tissue", "Lung Squamous Cell Carcinoma"]

# Preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_bytes):
    # Load image
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # First, predict Organ
        output_organ = model_organ(input_batch)
        prob_organ = torch.nn.functional.softmax(output_organ[0], dim=0)
        conf_organ, pred_idx_organ = torch.max(prob_organ, 0)
        conf_organ_val = float(conf_organ.item())
        
        if conf_organ_val < 0.7:
            return {"error": "Uncertain image, please upload a clearer sample"}
            
        organ_name = ORGAN_CLASSES[pred_idx_organ.item()]
        
        # Then, route to disease model based on organ
        if organ_name == "colon":
            output_colon = model_colon(input_batch)
            prob_colon = torch.nn.functional.softmax(output_colon[0], dim=0)
            conf_colon, pred_idx_colon = torch.max(prob_colon, 0)
            prediction = CLASS_NAMES_COLON[pred_idx_colon.item()]
            confidence = float(conf_colon.item())
        else:
            output_lung = model_lung(input_batch)
            prob_lung = torch.nn.functional.softmax(output_lung[0], dim=0)
            conf_lung, pred_idx_lung = torch.max(prob_lung, 0)
            prediction = CLASS_NAMES_LUNG[pred_idx_lung.item()]
            confidence = float(conf_lung.item())
            
    return {
        "organ": organ_name,
        "prediction": prediction,
        "confidence": confidence,
        "message": "Prediction successful"
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return JSONResponse(status_code=400, content={"message": "File is not an image"})
    
    contents = await file.read()
    results = predict_image(contents)
    
    if "error" in results:
        return JSONResponse(status_code=400, content={"message": results["error"]})
        
    return results

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Cancer Prediction API."}
