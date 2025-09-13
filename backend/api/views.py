import os, json, io
import torch, threading
import base64, json, os, random, mimetypes
from pathlib import Path
from PIL import Image
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from transformers import AutoTokenizer
from ml.pipelines.inference_pipeline import load_model_from_hopsworks_model_registry, inference
from ml.pipelines.inference_pipeline import FusionTransformerModel, ImageEncoderCNN, TextEncoderTransformer

torch.serialization.add_safe_globals([FusionTransformerModel])
torch.serialization.add_safe_globals([ImageEncoderCNN])
torch.serialization.add_safe_globals([TextEncoderTransformer])




@api_view(["GET"])
def get_items(request):
    return Response([{"name": "Item 1"}, {"name": "Item 2"}])  

# GLOBAL VARS
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

MODEL_NAME = "fusion_model_T5"      # check hopsworks, hopsworks model name
VERSION = None                      # if none gets latest
PROJECT = "medical_ml_project"      # hopsworks project

def _get_model_bundle():

    if _MODEL_CACHE["bundle"] is None:

        _MODEL_CACHE["bundle"] = load_model_from_hopsworks_model_registry(MODEL_NAME, VERSION, PROJECT)
    return _MODEL_CACHE["bundle"]


# ------------------------------- ROUTES -------------------------------


# ROUTE: call inference on user subitted image + patient details
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
    # model_bundle = _get_model_bundle()
    model_bundle = get_model_bundle_pickle() 
    

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





# ROUTE: Loads random sample 
@api_view(["POST"])
def load_random_sample_view(request):
    return Response({"ok": True})
    try:
        BASE_DIR = Path(settings.BASE_DIR)
    except Exception:
        BASE_DIR = Path(__file__).resolve().parents[2] 
    SAMPLE_IMAGES_DIR = Path(os.getenv("sample_images", "sample_images/"))
    SAMPLE_DETAILS_JSON = Path(os.getenv("sample_details", "sample_details/patient_details.json"))
    if not SAMPLE_IMAGES_DIR.exists():
        return Response(
            {"error": f"Images dir not found: {SAMPLE_IMAGES_DIR.resolve()}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    details_map = {}
    if SAMPLE_DETAILS_JSON.exists():
        try:
            details_map = json.loads(SAMPLE_DETAILS_JSON.read_text(encoding="utf-8"))
        except Exception as e:
            return Response({"error": f"Failed to read details JSON: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    exts = {".png", ".jpg", ".jpeg"}
    candidates = [p for p in SAMPLE_IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not candidates:
        return Response({"error": f"No images found in {SAMPLE_IMAGES_DIR.resolve()}."}, status=status.HTTP_404_NOT_FOUND)

    with_details = [p for p in candidates if p.name in details_map]
    pool = with_details if with_details else candidates

    chosen = random.choice(pool)
    image_bytes = chosen.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    mime = mimetypes.guess_type(chosen.name)[0] or "image/png"

    patient_details = details_map.get(
        chosen.name,
        "Age/sex, symptoms (onset/duration), key history, recent surgery/hospitalization, meds/O2, vitals, clinical question."
    )

    return Response(
        {
            "image_name": chosen.name,
            "image_mime": mime,
            "image_base64": b64,
            "patient_details": patient_details,
        },
        status=status.HTTP_200_OK,
    )



_BUNDLE = None
_LOCK   = threading.Lock()
# gets the path of where the bundle is locally
def _default_bundle_path() -> Path:
    # points to backend/ml/model/model_bundle.pt relative to this file
    try:
        from django.conf import settings
        base = Path(settings.BASE_DIR)
    except Exception:
        base = Path(__file__).resolve().parents[2]   # .../backend
    return (base / "ml" / "model" / "model_bundle.pt").resolve()

def _load_blob(src):
    if isinstance(src, (str, os.PathLike, Path)):
        p = Path(src)
        if not p.is_file():
            raise FileNotFoundError(f"Bundle not found: {p}")
        return torch.load(str(p), map_location="cpu")  # weights_only=True by default in 2.6
    if isinstance(src, (bytes, bytearray)):
        return torch.load(io.BytesIO(src), map_location="cpu")
    if hasattr(src, "read"):  # file-like (e.g., S3 StreamingBody)
        data = src.read()
        return torch.load(io.BytesIO(data), map_location="cpu")
    raise TypeError(f"Unsupported bundle source type: {type(src)}")

# Gets the model-bundle-pickel and reconstructs the bundle 
def get_model_bundle_pickle(path: str | os.PathLike | None = None):
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    with _LOCK:
        if _BUNDLE is not None:
            return _BUNDLE

        bundle_path = path or os.getenv("CXR_BUNDLE_PATH", _default_bundle_path())
        blob = _load_blob(bundle_path)

        # validate
        required = {"cfg", "fusion_state", "image_state", "text_state",
                    "t5_tokenizer_name", "bert_tokenizer_name"}
        missing = required.difference(blob.keys())
        if missing:
            raise ValueError(f"Bundle missing keys: {missing}")

        cfg = blob["cfg"]
        f   = (cfg.get("fusion") or {})
        d_img        = f.get("d_img", 1024)
        d_txt        = f.get("d_txt", 512)
        d_fuse_h     = f.get("d_fuse_hidden", 1024)
        n_disease    = f.get("n_disease", 13)
        n_cond_tokens= f.get("n_cond_tokens", 4)
        hf_t5_name   = (cfg.get("report_head") or {}).get("hf_model_name", "t5-small")

        # rebuild fresh modules, then load state dicts
        fusion = FusionTransformerModel(
            d_img=d_img, d_txt=d_txt, n_disease=n_disease,
            d_fuse_hidden=d_fuse_h, n_cond_tokens=n_cond_tokens,
            model_name=hf_t5_name, init_t5_from_config=True,
        )
        fusion.load_state_dict(blob["fusion_state"]); fusion.eval()

        image_encoder = ImageEncoderCNN()
        if blob["image_state"]:
            image_encoder.load_state_dict(blob["image_state"])
        image_encoder.eval()

        text_encoder = TextEncoderTransformer(
            model_name=(cfg.get("text_encoder") or {}).get("hf_model_name", "bert-base-uncased"),
            d_txt=d_txt, n_disease=n_disease
        )
        if blob["text_state"]:
            text_encoder.load_state_dict(blob["text_state"])
        text_encoder.eval()

        t5_tok   = AutoTokenizer.from_pretrained(blob["t5_tokenizer_name"])
        bert_tok = AutoTokenizer.from_pretrained(blob["bert_tokenizer_name"])

        artifacts = (cfg.get("artifacts") or {})
        class_names = artifacts.get("class_names", [
            "No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity",
            "Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis",
            "Pneumothorax","Pleural Effusion","Pleural Other","Fracture"
        ])
        thresholds = artifacts.get("thresholds", [0.5]*n_disease)

        _BUNDLE = {
            "cfg": cfg,
            "fusion_model": fusion,
            "image_encoder": image_encoder,
            "text_encoder": text_encoder,
            "t5_tok": t5_tok,
            "bert_tok": bert_tok,
            "class_names": class_names,
            "thresholds": thresholds,
            "version":999           # BUG: version im not able to get
        }
        return _BUNDLE

def clear_model_bundle():
    global _BUNDLE
    with _LOCK:
        _BUNDLE = None




# IMPORTANT
# from django.views.generic import TemplateView

# class ReactAppView(TemplateView):
#     template_name = 'index.html' 