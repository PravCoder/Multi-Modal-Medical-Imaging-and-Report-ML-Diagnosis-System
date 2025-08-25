# TRAINING PIPELINE: 
import hopsworks
import pandas as pd
import numpy as np
import json
from helper import print_clean_df
import warnings
warnings.filterwarnings("ignore", category=UserWarning)     # supress hopsworks warings for now caution!


# Global variables in pipeline
HOPS_PROJECT_NAME="medical_ml_project"
HOPS_FEATURE_GROUP_NAME="cxr_features"      # name of feature-group in feature-store where our data is
VERSION = 1
INPUT_COLS = []
OUTPUT_COLS = []

# Helper functions


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

# Transform them to be inputted into models

# Create models

# Train

# Save models to model registry


def training_tests():
    print("----------LOAD FEATURES/LABELS FROM FEATURE STORE HOPSWORKS---------")
    features_labels_df = load_features_labels_from_feature_store()
    print_clean_df(features_labels_df, num_rows=10)


training_tests()