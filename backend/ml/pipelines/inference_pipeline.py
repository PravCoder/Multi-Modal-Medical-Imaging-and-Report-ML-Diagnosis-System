import os, io, json, torch
import numpy as np
from typing import Dict, List
from PIL import Image
import hopsworks
from transformers import AutoTokenizer

from training_pipeline import ImageEncoderCNN
from training_pipeline import TextEncoderTransformer
from training_pipeline import FusionTransformerModel
from training_pipeline import image_transfom_into_tensor


# given a model-registry & model-name findslatest version in model in that registry
def latest_version(model_reg, name: str) -> int:
    models = model_reg.get_models(name=name)
    if not models:
        raise RuntimeError(f"No models named '{name}' found.")
    return max(m.version for m in models)


"""
Download a model version directory from Hopsworks and rebuild:
    - FusionTransformerModel (+ weights)
    - ImageEncoderCNN / TextEncoderTransformer (+ weights, if saved)
    - T5 tokenizer from t5_assets or HF hub
    - BERT tokenizer for patient details
    - Class names and thresholds (if present)
Returns a bundle dict with everything needed for inference.
"""
def load_model_from_hopsworks_model_registry(model_name, version=None, project_name=None):
    # logins to hopsworks project
    project = hopsworks.login(project=project_name) if project_name else hopsworks.login()
    # gets model-registry reference associated with that project
    model_registry = project.get_model_registry()

    if version is None:
        version = latest_version(model_registry, model_name)    # get the version of model in registry
    
    # get the model in registry
    registry_model = model_registry.get_model(model_name, version=version)
    # download folder in model registry with fusion_model.pt, config.json, t5_assets/, etc.
    local_dir = registry_model.download()  
    # load config & artifacts
    config_path = os.path.join(local_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)  # config-dict we uploaded with all the attributes/data

    # Get fusion shapes, with fallbacks, fusion-tranformer-model-attributes, same dict-keys has as configuration-dict in save_model_to_hopsworks_model_registry()
    fusion_cfg = cfg.get("fusion", {}) or {}
    d_img = fusion_cfg.get("d_img", 1024)
    d_txt = fusion_cfg.get("d_txt", 1024)
    d_fuse_hidden = fusion_cfg.get("d_fuse_hidden", 1024)
    n_disease = fusion_cfg.get("n_disease", 13)
    n_cond_tokens = fusion_cfg.get("n_cond_tokens", 4)

    # T5-report-head-HF-transformer
    report_head_cfg = cfg.get("report_head", {}) or {}  # get the report-head-key (which is its own dict) from configuration
    t5_saved_dir = os.path.join(local_dir, "t5_assets") # get t5-assests
    has_t5_assets = os.path.isdir(t5_saved_dir) and os.path.exists(os.path.join(t5_saved_dir, "config.json"))   # get config.json
    gen_model_name = t5_saved_dir if has_t5_assets else report_head_cfg.get("hf_model_name", "t5-small") # get the report-head model name

    # Get BERT tokenizer name from configuration dict
    bert_name = bert_name = (cfg.get("text_encoder", {}) or {}).get("hf_model_name", "bert-base-uncased")

    # get some stuff from
    t5_dir = os.path.join(local_dir, "t5_assets")
    has_t5_assets = os.path.isdir(t5_dir) and os.path.exists(os.path.join(t5_dir, "config.json"))
    hf_t5_name = (cfg.get("report_head", {}) or {}).get("hf_model_name", "t5-small")

    # Given the attributes we extract from configuration-dict create the fusion-tranformer-model-obj, make sure to pass in init_t5_from_config, t5_assets_dir into fusion-constructor 
    fusion_model = FusionTransformerModel(d_img=d_img, d_txt=d_txt, n_disease=n_disease, d_fuse_hidden=d_fuse_hidden, model_name=gen_model_name, n_cond_tokens=n_cond_tokens, init_t5_from_config=not has_t5_assets, t5_assets_dir=t5_dir if has_t5_assets else None )
    # loads serialized dict of tensors for all parameters & copy those tensors into the initialized fusion-obj, .pt is the file we uploaded to model-registry, 
    fusion_model.load_state_dict(torch.load(os.path.join(local_dir, "fusion_model.pt"), map_location="cpu"))    # load *our* fine-tuned weights this includes (this includes T5 submodule params)  if save_t5_weights was set to false
    fusion_model.eval()

    image_encoder = None
    img_sd = os.path.join(local_dir, "image_encoder.pt")    # load the image-encoder.pt file we uploaded to hopsworks model registry
    if os.path.exists(img_sd):
        image_encoder = ImageEncoderCNN()           
        # load serialized dict of tensors of all parameters & copy those tensors into initialized image-encoder-obj
        image_encoder.load_state_dict(torch.load(img_sd, map_location="cpu"))
        image_encoder.eval()

    text_encoder = None
    txt_sd = os.path.join(local_dir, "text_encoder.pt") # load text-encoder.pt file we uploaded to hopsworks
    if os.path.exists(txt_sd):
        # load serialized dict of tensors fo all parameters & copy those tensors into initialized text-encoder-obj
        text_encoder = TextEncoderTransformer(model_name=bert_name, d_txt=d_txt, n_disease=n_disease)
        text_encoder.load_state_dict(torch.load(txt_sd, map_location="cpu"))
        text_encoder.eval()

    # recreate the tokenizers
    t5_tok   = AutoTokenizer.from_pretrained(gen_model_name)   # for decoding generated report
    bert_tok = AutoTokenizer.from_pretrained(bert_name)        # for patient-details

    # get the artifcats from configuration we passed in 
    artifacts = cfg.get("artifacts", {}) or {}
    class_names = artifacts.get("class_names", [
        "No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity",
        "Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis",
        "Pneumothorax","Pleural Effusion","Pleural Other","Fracture"
    ])
    thresholds = artifacts.get("thresholds", [0.5] * n_disease)

    return {
        "dir": local_dir,   # folder in model-registry with all the .pt files
        "version": version,     # version that was passed in
        "cfg": cfg,             # configuration-dict we uploaded tor registry
        "fusion_model": fusion_model,   # fusion-tranformer-obj we reconstructed
        "image_encoder": image_encoder, # image-encoder-obj we reconstructed
        "text_encoder": text_encoder,   # test-encoder-obj we reconstructed
        "t5_tok": t5_tok,               # t5-tokenizer we recreated
        "bert_tok": bert_tok,           # vert-tokenizer we recreated
        "class_names": class_names,     # artifact, check config we passed
        "thresholds": thresholds,       # artifact
    }

def inference_tests():
    print("\n------------LOAD & RECONSTRUCT MODEL FROM HOPSWORKS MODEL REGISTRY------------")
    MODEL_NAME = "fusion_model_T5"      # check hopsworks
    VERSION = None                      # if none gets latest
    PROJECT = "medical_ml_project"      # hopsworks project

    model_bundle = load_model_from_hopsworks_model_registry(MODEL_NAME, VERSION, PROJECT)
    print(f"model_bundle loaded from registry: \n{model_bundle.keys()}" )


    print("\n------------INFERENCE (SINGLE or BATCH)------------")



if __name__ == "__main__":
    inference_tests()
