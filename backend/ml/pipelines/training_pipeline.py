# TRAINING PIPELINE: 
import hopsworks
import pandas as pd
import numpy as np
import json
from helper import print_clean_df
import torch
import warnings
# import tensorflow as tf
import io
import os
import boto3
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)     # supress hopsworks warings for now caution!


# Global variables in pipeline
HOPS_PROJECT_NAME="medical_ml_project"
HOPS_FEATURE_GROUP_NAME="cxr_features"      # name of feature-group in feature-store where our data is
VERSION = 1
INPUT_COLS = []
OUTPUT_COLS = []
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

# Transform them to be inputted into models

# Create models

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



training_tests()