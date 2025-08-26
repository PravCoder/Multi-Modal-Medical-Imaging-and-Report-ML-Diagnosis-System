# TRAINING PIPELINE: 
import hopsworks
import pandas as pd
import numpy as np
import json
from helper import print_clean_df
import warnings
import io
from PIL import Image
import boto3
import torch
import torch.nn as nn            # neural network layers modules
import torchvision.models as tv  # ready-made CNN backbones (ResNet, etc
import torchvision.transforms as T
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)     # supress hopsworks warings for now caution!


# Global variables in pipeline
HOPS_PROJECT_NAME="medical_ml_project"
HOPS_FEATURE_GROUP_NAME="cxr_features"      # name of feature-group in feature-store where our data is
VERSION = 1
INPUT_COLS = []
OUTPUT_COLS = []
IMG_SIZE = 224
s3 = boto3.client("s3")



# Get features & labels from hopsworks feature-group in feature store
def load_features_labels_from_feature_store():
    # connects local python env to hopsworks
    project = hopsworks.login(project=HOPS_PROJECT_NAME)
    # gets feature store reference associated with the hopsworks-project
    fs = project.get_feature_store()
    # gets the feature-group in feature-store "fs"
    fg = fs.get_feature_group(name=HOPS_FEATURE_GROUP_NAME, version=VERSION)

    select_columns = ["image_url", "patient_details", "disease_classification_vector", "report", "event_time"]
    as_of = None    
    # build a query and read select_all() df
    q = fg.select_all() if not select_columns else fg.select(select_columns)
    features_labels_df = (q.as_of(as_of).read() if as_of else q.read())

    # over time if you have multiple rows for same image-url (which we preivously get as primary key), keep the latest event-time if present
    if "event_time" in features_labels_df.columns:
        features_labels_df = (features_labels_df.sort_values("event_time").groupby("image_url", as_index=False).tail(1).reset_index(drop=True))

    return features_labels_df

# --------Some helper functions to get images from S3-bucket--------
def get_image_from_s3(bucket, key):  # gets the image from s3 given bucket-name & objects key
    bio = io.BytesIO()                      # creates in-memory binary stream which is an object that behaves as a file, but doesnt interact with file system
    s3.download_fileobj(bucket, key, bio)   # downloads object from s3 as a file like object, given bucket and key of object
    return bio.getvalue()                   # returns the entire contents of in-mem buffer as a bytes object, this is the image file as a bytes object retrived fomr s3

# given a s3-url of an image gets the bucket and key of where that image lives, so we can get that image file
def parse_s3_url(url):
    assert url.startswith("s3://")
    no_schema = url[5:]
    bucket, key = no_schema.split("/", 1)
    return bucket, key

# ===========================================================
# Image Encoder
# ===========================================================

# this is a torchvision transform pipeline that accepts a PIL Image object
# it returns a torch.FloatTensor of shape [3, IMG_SIZE, IMG_SIZE] for three color channels because ImageNet is like that. 
# this is applied to each image individually and stacked the batch shape is [B, 3, IMG_SIZE, IMG_SIZE]
image_transfom = T.Compose([
    T.Resize(256, antialias=True),    # takes in pil-image, resizes so shorter side is 256-pixels
    T.CenterCrop(IMG_SIZE),           # crops the center square out of the image (224, 224, C)
    T.ToTensor(),                     # converts pil -> torch.FloatTensor, reorders dims to channel dim first [C, H, W], If grayscale input → [1, 224, 224]. If RGB input → [3, 224, 224], scale pixel values to [0,1]
    T.Lambda(lambda x: x.repeat(3,1,1) if x.size(0) == 1 else x),  # ensures 3 channels for colors, If input has C=1 (grayscale), repeats that channel 3 times: [1,224,224] → [3,224,224]. If already C=3, leaves unchanged.
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), # normalizes each channel seperately
    # output: [3, 224, 224] tensor representing image to be fed into cnn-image-encoder, 
])


# ===========================================================
# Text Encoder
# ===========================================================

# ===========================================================
# Fusion Model
# ===========================================================

# Train

# Save models to model registry


def training_tests():
    print("----------LOAD FEATURES/LABELS FROM FEATURE STORE HOPSWORKS---------")
    # features_labels_df = load_features_labels_from_feature_store()
    # print_clean_df(features_labels_df, num_rows=10)


    print("---------GET IMAGE FROM S3 AS BYTES OBJECT---------")
    s3_image_url_example = "s3://medical-ml-proj-bucket/chest-x-ray-images/cc4ff72b-a3adf0.jpg"
    bucket, key = parse_s3_url(s3_image_url_example)
    image_bytes_obj = get_image_from_s3(bucket, key)
    print(f"image bytes object: {image_bytes_obj[0:5]}")

    print("----------IMAGE ENCODER: IMAGE TRANSFORM TEST SINGLE IMAGE")
    pil_img = Image.open(io.BytesIO(image_bytes_obj))       # convert iamge-bytes intopil-img-obj
    tensor_img = image_transfom(pil_img)                    # convert pil-img into tensor format to be fed into cnn image encoder
    print(tensor_img.shape)  # should be [3, 224, 224], [3, img_sze, img_sze], check image-size constant
    print(tensor_img.dtype) 




training_tests()