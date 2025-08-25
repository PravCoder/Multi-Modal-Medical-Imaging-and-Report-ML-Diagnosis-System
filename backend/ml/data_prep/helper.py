import random
import re

# order must match your disease classification vector:
DISEASES = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture"]

# symptom pool by disease broad, non-diagnostic
SYMPTOMS_MAP = {
    "No Finding": ["asymptomatic, routine screening"],
    "Enlarged Cardiomediastinum": ["chest discomfort", "shortness of breath on exertion", "fatigue"],
    "Cardiomegaly": ["fatigue", "breathlessness on exertion", "swelling of ankles"],
    "Lung Opacity": ["cough", "shortness of breath", "low grade fever"],
    "Lung Lesion": ["chronic cough", "weight loss", "chest pain"],
    "Edema": ["difficulty breathing when lying down", "nighttime breathlessness", "leg swelling"],
    "Consolidation": ["productive cough", "fever", "pleuritic chest pain", "shortness of breath"],
    "Pneumonia": ["fever", "productive cough", "pleuritic chest pain", "malaise"],
    "Atelectasis": ["shortness of breath", "chest discomfort", "dry cough"],
    "Pneumothorax": ["sudden chest pain", "acute shortness of breath"],
    "Pleural Effusion": ["shortness of breath", "pleuritic chest pain", "dry cough"],
    "Pleural Other": ["pleuritic chest pain", "chest tightness", "shortness of breath"],
    "Fracture": ["localized chest wall pain", "tenderness", "pain with deep breathing"],
}

def _sample_bool(p):  
    return random.random() < p

def _pack_years(smoker):
    if not smoker:
        return None
    # skewed but simple: light 5–10, moderate 10–20, heavy 20–40
    r = random.random()
    if r < 0.4:  return random.randint(5, 10)
    if r < 0.8:  return random.randint(11, 20)
    return random.randint(21, 40)

def _infer_hints(text):
    """Very light hinting from report text (optional)."""
    t = (text or "").lower()
    hints = set()
    if re.search(r'\bfever|febrile|pyrexia\b', t): hints.add("fever")
    if re.search(r'\bcough|sputum\b', t): hints.add("cough")
    if re.search(r'\bpleur|chest pain\b', t): hints.add("pleuritic chest pain")
    if re.search(r'\bdyspn(ea|o(e)?a)|shortness of breath|sob\b', t): hints.add("dyspnea")
    if re.search(r'\bedema|effusion|orthopnea\b', t): hints.add("leg swelling")
    return hints


