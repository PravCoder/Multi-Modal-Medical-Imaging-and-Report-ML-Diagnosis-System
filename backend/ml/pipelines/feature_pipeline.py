# FEATURE PIPELINE: transforms raw data into ready-to-train features/labels saved in feature store, this just cleans the raw data and saves it in our feature store hopsworks, the training pipeline transforms that into ready-to-train tensors
import pandas as pd
import numpy as np
import json
import hopsworks
from dotenv import load_dotenv
# from helper import print_clean_df. # not working dont have time
def print_clean_df(df, num_rows=5, display_head=True, str_length=15):
    display_df = df.copy()
    rows = df.shape[0]
    cols = df.shape[1]

    # Get the columns that contain string data
    str_cols = display_df.select_dtypes(include=['object']).columns
    # Truncate string columns
    for col in str_cols:
        display_df[col] = display_df[col].astype(str).str.slice(0, str_length) + \
                          display_df[col].astype(str).str.len().gt(str_length).apply(lambda x: '...' if x else '')
    # Print the head or tail
    if display_head:
        print(display_df.head(num_rows).to_string())
        print(f"Number of rows: {rows}")
        print(f"Number of columns: {cols}")
    else:
        print(display_df.tail(num_rows).to_string())
        print(f"Number of rows: {rows}")
        print(f"Number of columns: {cols}")

load_dotenv()


# Global Variables in pipeline
RAW_DATA_COLUMNS = ["image_url", "patient_details", "disease_classification_vector", "report"]

# Helper functions



# Enforces data-types in raw-data-df
def enforce_raw_data_columns(df):
    # make sure image-url input col, patient-detials input-col, and report output col are of type string in dataframe
    df["image_url"] = df["image_url"].astype(str)
    df["patient_details"] = df["patient_details"].astype(str)
    df["report"] = df["report"].astype(str)

    def _to_float_vector(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
        else:
            arr = np.asarray(json.loads(str(x)), dtype=float)
        if arr.size != 13: # for 13 disease classes
            raise ValueError(f"Expected vector of length 14, got {arr.size}")
        return arr

    # make sure disease-class-vec [1,0,0,1,0,1,...] is of type float vector
    df["disease_classification_vector"] = df["disease_classification_vector"].map(_to_float_vector)
    return df

# Load in raw data from S3-bucket given s3-bucket-url, return pandas dataframe of raw data
def load_raw_data(s3_url):
    # load parquet raw-data-file from s3-bucket using s3-url of where the parquert file is located, and convert it into a pandas-dataframe
    raw_data_df = pd.read_parquet(s3_url, columns=RAW_DATA_COLUMNS, storage_options={}, engine="pyarrow")
    # make sure columns in raw data are correct type
    raw_data_df = enforce_raw_data_columns(raw_data_df)
    return raw_data_df


FEATURE_GROUP_NAME="cxr_features"
# Save raw-data-cleaned df in hopsworks feature-store
def save_cleaned_raw_data_to_feature_store(cleaned_df):
    cleaned_df["event_time"] = pd.Timestamp.now()   # just add an even-time column for when this event happened which is current time
    project = hopsworks.login()     # connects local python environment to hopsworks projects

    # gets reference to feature store assoicated with the hopsworks project medical_ml_project
    fs = project.get_feature_store()        

    # this either gets or creates a feature group which is a subset of a feature-store, feature-group called cxr_features
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,            # the feature group cxr-features is the dataframe that has both cleaned-features/labels in feature stored
        version=1,
        primary_key=["image_url"],
        event_time="event_time",
        online_enabled=True,   
    )

    # update existing feature-group with new data from thie given dataframe, false if you dont want to stop in terminal and want to return to code if job is not done.
    fg.insert(cleaned_df, write_options={"wait_for_job": False})



def feature_pipeline_tests():
    
    print("\n---------LOAD SHOW RAW DATA:---------")
    cleaned_raw_data_df = load_raw_data("s3://medical-ml-proj-bucket/raw_data/dataset.parquet")       # make sure this url of where the raw data is in s3 is correct
    print_clean_df(cleaned_raw_data_df, num_rows=3)
    # this already does some cleaning in the load func, but most of the transformation is done in training-pipeline before training into tensors thats why we just pass the cleaned-raw-data-df into save func
    print(f"Number of rows/examples & features/labels : {cleaned_raw_data_df.shape}")

    print("\n---------SAVE CLEANED RAW DATA TO HOPSWORKS FEATURE STORE---------")
    save_cleaned_raw_data_to_feature_store(cleaned_raw_data_df)

    # Images stored in S3, dataset sotred in hopsworks


if __name__ == "__main__":

    feature_pipeline_tests()

