import os, json
import torch
from PIL import Image
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework import status
from ml.pipelines.inference_pipeline import load_model_from_hopsworks_model_registry, inference

@api_view(["GET"])
def get_items(request):
    return Response([{"name": "Item 1"}, {"name": "Item 2"}])  


_MODEL_CACHE = {"bundle": None}

DISEASES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
]
MODEL_NAME = "fusion_model_T5"      # check hopsworks
VERSION = None                      # if none gets latest
PROJECT = "medical_ml_project"      # hopsworks project

def _get_model_bundle():

    if _MODEL_CACHE["bundle"] is None:

        _MODEL_CACHE["bundle"] = load_model_from_hopsworks_model_registry(MODEL_NAME, VERSION, PROJECT)
    return _MODEL_CACHE["bundle"]


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def predict_view(request):
    # 1) ------Validate inputs------
    image_file = request.FILES.get("image")
    patient_details = request.data.get("patient_details", "")

    if not image_file:
        return Response({"error": "Missing 'image' file."}, status=status.HTTP_400_BAD_REQUEST)
    try:
        image_pil = Image.open(image_file).convert("RGB")
    except Exception:
        return Response({"error": "Invalid image format."}, status=status.HTTP_400_BAD_REQUEST)

    # 2) ------Load cached model bundle-----
    model_bundle = _get_model_bundle()
    

    # 3) ------Run inference------
    # Expected return (example):
    # {
    #   "disease_probs": 
    #   "report_text":
    # }
    preds = inference(model_bundle, image_pil, patient_details)
    print(f"Inference predictions: {preds}")

    raw_probs = preds.get("disease_probs") or preds.get("disease_probs") or {}
    report_text = preds.get("report_text", {}) or {}

    # 4) -----Normalize to frontend shape (percentages 0-100, 2dp)-----
    diseases = []
    for name in DISEASES:
        p = float(raw_probs.get(name, 0.0))
        if p <= 1.0:
            p *= 100.0
        diseases.append({"name": name, "probability": round(p, 2)})

    payload = {
        "diseases": diseases,
        "report_text": report_text,
    }
    return Response(payload, status=status.HTTP_200_OK)

