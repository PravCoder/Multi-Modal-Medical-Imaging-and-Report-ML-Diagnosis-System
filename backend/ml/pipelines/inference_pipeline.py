# RUN SCRIPT IMPORTS: make sure your are in root project
# import os, io, json, torch
# import numpy as np
# from typing import Dict, List
# from PIL import Image
# import hopsworks
# from transformers import AutoTokenizer
# from training_pipeline import ImageEncoderCNN
# from training_pipeline import TextEncoderTransformer
# from training_pipeline import FusionTransformerModel
# from training_pipeline import image_transfom_into_tensor, tokenize_patient_details
# from training_pipeline import parse_s3_url, get_image_from_s3

# RUN APP IMPORTS: make sure your are in backend
import os, io, json, torch
import numpy as np
from typing import Dict, List
from PIL import Image
import hopsworks
from transformers import AutoTokenizer
from .training_pipeline import ImageEncoderCNN
from .training_pipeline import TextEncoderTransformer
from .training_pipeline import FusionTransformerModel
from .training_pipeline import image_transfom_into_tensor, tokenize_patient_details
from .training_pipeline import parse_s3_url, get_image_from_s3


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

"""
Returns:
      {
        report_text: str,
        disease_probs: {class_name: float},
        disease_vector: [0/1]*13,
        model_version: int
      }
"""
@torch.no_grad()        
def inference(model_bundle, image_pil, patient_details, device=None, gen_kwargs=None):
    if isinstance(device, torch.device):
        dev = device
    elif isinstance(device, str):
        dev = torch.device(device)
    elif device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise TypeError(f"'device' must be str|torch.device|None, got {type(device)}")


    # just get the keys from the bundle-dict
    fusion_nodel = model_bundle["fusion_model"].to(dev)
    t5_tok = model_bundle["t5_tok"]
    bert_tok = model_bundle["bert_tok"]
    class_names = model_bundle["class_names"]
    thresholds = torch.tensor(model_bundle["thresholds"], device=dev)
    
    # get the imagee-encoder-cnn-model & text-encoder-tranfromer-model from bundle
    image_encoder = model_bundle["image_encoder"].to(dev).eval()
    text_encoder  = model_bundle["text_encoder"].to(dev).eval()

    # ----- preprocess inputs -----
    x_img = image_transfom_into_tensor(image_pil).unsqueeze(0) .to(dev)  # transform the image into tensor, add batch dim, [1,3,224,224] [B, RGB, H, W]
    tok = tokenize_patient_details([patient_details], max_len=96)  # takes in patient-details string and returns dict where ["input_ids"] it spltis the string into tokens and gives each token and ID
    tok = {k: v.to(dev) for k, v in tok.items()}           # dict of [1,L]

    # ----- Forward pass encoders -----
    z_img = image_encoder(x_img)["embeddings"]  # calling image-cnn-encoder passing tensor of image-input, returns dict with embedding-key equal to [1, D_img] tensor-embedding representing that image
    z_txt = text_encoder(**tok)["embeddings"]   # calling text-transformer-encoder passing tokens of patient-details-input, returns dict with embedding-key equal to [1, D_txt] tensor-embedding-vector representing that text, B=1 batch for inference

    # ----- Disease head output  -----
    out = fusion_nodel(z_img=z_img, z_txt=z_txt, report_labels=None)    # call forward-pass of fusion-model returns logits disease classificaiton vector
    logits = out["disease_logits"]                              # [1, N]
    probs  = torch.sigmoid(logits)[0]                           # [N]
    vector = (probs >= thresholds).int().tolist()
    
    # ----- Report generatio output -----
    # list some attributes for generation, min number of tokens we want
    gen_attributes = dict(max_new_tokens=180, min_new_tokens=150, num_beams=4, no_repeat_ngram_size=3,length_penalty=1.1, early_stopping=True,eos_token_id=t5_tok.eos_token_id, pad_token_id=t5_tok.pad_token_id)
    if gen_kwargs: gen_attributes.update(gen_kwargs)
    # call fusion-model generate-func pass ing image-embedding from image-encoder, text-embedding from text-emcoder
    # returns T5 tokenizer tokens IDs for each token in the generated text from the T5 tokenizer, [1, L_gen]
    gen_ids = fusion_nodel.generate(z_img, z_txt, **gen_attributes)  
    # given gen-ids a tensor of token IDS for that patient-details-string coverts back to text using the same T5 tokenize    
    report  = t5_tok.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # return dict of outputs
    return {
        # generated-text-output
        "report_text": report,  
        "disease_probs": {class_names[j]: float(probs[j]) for j in range(len(class_names))},
        # disease-vector
        "disease_vector": vector,
        "model_version": model_bundle["version"],
    }

    


def inference_tests():
    print("\n------------LOAD & RECONSTRUCT MODEL FROM HOPSWORKS MODEL REGISTRY------------")
    MODEL_NAME = "fusion_model_T5"      # check hopsworks
    VERSION = None                      # if none gets latest
    PROJECT = "medical_ml_project"      # hopsworks project

    model_bundle = load_model_from_hopsworks_model_registry(MODEL_NAME, VERSION, PROJECT)
    print(f"model_bundle loaded from registry: \n{model_bundle.keys()}" )


    print("\n------------INFERENCE (SINGLE or BATCH)------------")
    # testing with an image in s3, in app user will upload image, we convert to pil-obj and pass to inference pipeline
    patient_details = "44 year old female PA view , hypertension , cough"
    image_url = "s3://medical-ml-proj-bucket/chest-x-ray-images/68574aaf-0dd83b.jpg"   
    bkt, ky = parse_s3_url(image_url)
    img_bytes = get_image_from_s3(bkt, ky)
    # PIL-object of image will be the input that the inference gets
    image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    output = inference(model_bundle, image_pil, patient_details)
    print(f"Inference output: {output}")
    


if __name__ == "__main__":
    inference_tests()
