# File: prepares the raw data before it is transformed in the feature pipeline. This is the "pre-preparation" since we do not have a single dataset for our needs. 
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
import json
import random
from helper import DISEASES, SYMPTOMS_MAP
import io, os, hashlib, uuid
import boto3
from PIL import Image
from datasets import Value
load_dotenv()




dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset", split="train")
diseases_classes = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture"]


def show_basic_info(dataset):
    print(f"Features: {dataset.features}")
    print(f"Number of Examples: {len(dataset)} rows.")

    # First row sample
    sample = dataset[0] 
    print(f"Sample Row: {sample}")  # dict with keys the rows names "image", "findings", "impressions"
    print(f"Image Shape: {sample["image"].size}")  # (width, height)

    # Iterate all images and make sure they can be opened
    for idx, row in enumerate(dataset):
        try:
            img = row["image"]  # already a PIL.Image object
            img.verify()        # verify image integrity
        except Exception as e:
            print(f"Error with image at index {idx}: {e}") # if no expection then all rows images can be opened

def show_example_image(sample):
    # View Image of an example, PIL.Image object
    img = sample["image"]   
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"Impression: {sample['impression']}\nFindings: {sample['findings']}")
    plt.show()




# Given an example puts its findings + impression column text into openAI and asks it to find the diagnosis, this will be the disease diagnosis target column.
# returns dict where key is disease classifciation and value is either 0/1 if it exists or not. 
"""
Input:
Lung volumes are low. This results in crowding of the bronchovascular structures. There may be mild pulmonary vascular congestion. 
Output:
{
  "No Finding": 0,
  "Enlarged Cardiomediastinum": 0,
  "Cardiomegaly": 1,
  "Lung Opacity": 1,
  "Lung Lesion": 1,
  "Edema": 1,
  "Consolidation": 0,
  "Pneumonia": 0,
  "Atelectasis": 0,
  "Pneumothorax": 0,
  "Pleural Effusion": 0,
  "Pleural Other": 0,
  "Fracture": 0
}
"""
def get_example_disease_classification(sample):
    client = OpenAI()
    diseases_classes = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture"]
    findings = sample["findings"]
    impression = sample["impression"]
    text = f"Findings: {findings}\nImpression: {impression}"

    prompt = f"""
You are a medical AI assistant. 
Given the following radiology report on a chest X-ray, classify whether each of these diseases/observations is present (1) or absent (0), mark it as 1 if it is mentioned or implied. No Finding should be 1 only if all other diseases are 0.

{', '.join(diseases_classes)}

Text:
{text}

Respond ONLY as a valid JSON object with the disease names as keys and 0/1 as values. If No Finding is 1 no other disease can be 1. 
    """

    resp = client.chat.completions.create(
        model="gpt-5", 
        messages=[{"role": "user", "content": prompt}],
    )

    # print(resp.choices[0].message.content)
    return json.loads(resp.choices[0].message.content)

# Given an example generates the disease classifciation dict for it and converts it into a vector and returns it for a single example
def generate_disease_vector(example):
    label_dict = get_example_disease_classification(example)    # dict where key is disease name and value is either 0/1 if that disease exists
    label_vector = [label_dict[d] for d in diseases_classes]
    example["disease_classification_vector"] = label_vector     # add column for disease classification that is the label vector [0,1, 0, 1, 1,...]
    return example






# Given an example in HF-dataset it generates the patient-detials-input column value for this example
def generate_patient_details_single_example(example, seed=None):
    """Generate synthetic patient details string given one dataset row."""
    if seed is not None:
        random.seed(seed)

    # --- demographics ---
    age = random.randint(18, 90)
    sex = random.choice(["male", "female"])
    view = random.choice(["AP", "PA"])

    # --- risk factors ---
    smoker = random.random() < 0.25
    diabetes = random.random() < 0.10
    hypertension = random.random() < 0.30

    risk_factors = []
    if smoker:
        pack_years = random.randint(5, 40)
        risk_factors.append(f"smoking history of {pack_years} pack years")
    if diabetes:
        risk_factors.append("diabetes")
    if hypertension:
        risk_factors.append("hypertension")

    # --- disease vector must exist in example ---
    vec = example["disease_classification_vector"]
    present = [d for d, v in zip(DISEASES, vec) if v == 1]

    # choose symptoms from map
    symptom_pool = []
    for d in present or ["No Finding"]:
        symptom_pool.extend(SYMPTOMS_MAP.get(d, []))

    if present == ["No Finding"]:
        chosen_symptoms = []
    else:
        k = random.randint(1, min(3, len(symptom_pool)))
        chosen_symptoms = random.sample(symptom_pool, k)

    # --- build details string ---
    parts = [f"{age} year old {sex}", f"{view} view"]
    if risk_factors:
        parts.append(", " + ", ".join(risk_factors))
    if chosen_symptoms:
        parts.append(", " + ", ".join(chosen_symptoms))
    else:
        parts.append(", routine evaluation")

    example["patient_details"] = " ".join(parts)
    return example

# just a wrapper so we can do dataset.map(create_patient_details_column) and aply it ao all examples
def create_patient_details_column(example):
    return generate_patient_details_single_example(example)




# given an example-row in HF-dataset just returns concat of findings +impression column, which is the value of the report output target column for this eample
def combine_findings_and_impression_cols(example):
    return {"report": example["findings"] + example["impression"]}




def tests():
    dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset", split="train")
    sample = dataset[2] 

    print("\n-----BASIC INFO + VALIDATE IMAGES-----:")
    # show_basic_info(dataset)

    print("\n-----ShOW IMAGE-----:")
    # show_example_image(sample)

    print("\n-----GET DISEASE CLASSIFICATION FROM A SINGLE EXAMPLE-----:")
    # print(f"Findings + Impression: {sample["findings"]} {sample["impression"]}")
    # disease_label_dict = get_example_disease_classification(sample)
    # print(disease_label_dict)

    
    print("\n-----CREATE DISEASE CLASSIFICATION OUTPUT COLUMN-----:")
    # we are taking the dataset and creating the target output column disease-classification based on the findings+impression columns of the raw data
    small_dataset = dataset.select([0, 1, 2])
    small_dataset = small_dataset.map(generate_disease_vector) # apply this function generate_disease_vector() to every example in dataset, so it creates the disease_classification_vector target column
    print(f"Example #1: {small_dataset[0]}")
    print(f"\nExample #2: {small_dataset[1]}")
    print(f"\nExample #3: {small_dataset[2]}")


    print("\n-----CREATE PATIENT DETAILS INPUT COLUMN-----:")
    # we are taking the dataset and creating the input column patient-detials using rule-based function
    small_dataset = small_dataset.map(create_patient_details_column)    # make sure this is same dataset as above with disease-classification column else it wont work, apply func to all examples in dataset
    print(f"Example #1: {small_dataset[0]}")
    print(f"\nExample #2: {small_dataset[1]}")
    print(f"\nExample #3: {small_dataset[2]}")


    print("\n-----CREATE REPORT OUTPUT COLUMN-----:")
    # we are taking the dataset and creating the output column report
    small_dataset = small_dataset.map(combine_findings_and_impression_cols, remove_columns=["findings", "impression"])  # apply func all examples in daataset and remove cols
    print(f"Example #1: {small_dataset[0]}")        # print a whole row

# tests()    - uncomment this



# ---------- Below Is Code For Saving The Raw Data To Aws ----------

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
s3 = session.client("s3")


# This takes image object from PIL library and turns it into a set of bytes representing a compressed JPEG image
def _pil_to_jpeg_bytes(pil_img, quality=95):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

# This calculates a hash of a binary data
def _digest(b: bytes):
    return hashlib.sha256(b).hexdigest()

# takes in a example-row in HF-dataset and uploads that image of that example into S3, and replaces the image column with the image's url in s3. 
def upload_image_and_create_image_url_column_single_example(example):
    img = example["image"]              # get the PIL.Image object in image-column of this example
    b = _pil_to_jpeg_bytes(img, 95)
    sha = _digest(b)[:8]                # get hash of this image
    key = f"chest-x-ray-images/{sha}-{uuid.uuid4().hex[:6]}.jpg"    # construct unique s3-key of where this image will be saved in bucket

    # uploads image to s3-bucket speciynig bucket-name, s3-object-key, body which is binary image to be uploaded
    s3.put_object(
        Bucket=os.getenv("AWS_S3_BUCKET_NAME"), Key=key, Body=b,
        ContentType="image/jpeg",
        ServerSideEncryption="AES256"    # or 'aws:kms' + SSEKMSKeyId=...
    )

    # replaces image column with image-url in s3-bucket which contains the object s3-key
    example["image_url"] = f"s3://{os.getenv("AWS_S3_BUCKET_NAME")}/{key}"
    return example

# takes in HF-ddataset after all pre-preparation and converts it into parquet-file and uploads it into s3-bucket based on env-vars
def save_dataset_as_parquet_in_s3(dataset):
    df = dataset.to_pandas()


    parquet_key = "raw_data/dataset.parquet"
    df.to_parquet(
        f"s3://{os.getenv("AWS_S3_BUCKET_NAME")}/{parquet_key}",
        index=False,
        storage_options={"key": os.getenv("AWS_ACCESS_KEY_ID"),
                        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
                        "client_kwargs": {"region_name": os.getenv("AWS_REGION")}}
    )
    print("Wrote:", f"s3://{os.getenv("AWS_S3_BUCKET_NAME")}/{parquet_key}")


def prepare_tests():
    # note these tests are before the pre-preparation of raw data
    print("\n----- SAVE IMAGES IN S3 & CREATE IMAGE-URL COLUMN -----:")
    dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset", split="train")
    small_dataset = dataset.select([0])
    small_dataset = small_dataset.map(upload_image_and_create_image_url_column_single_example)
    # remove the old image column because it was of type image object, and add new col image-url which is the location where the image is stored
    small_dataset = small_dataset.map(upload_image_and_create_image_url_column_single_example, remove_columns=["image"])
    print(f"Single row: {small_dataset[0]}")    # make sure image-url col is there


    print("\n----- SAVE DATASET AS PARQUET IN S3 -----:")
    save_dataset_as_parquet_in_s3(small_dataset)    # you can view parquet-file in s3
    
# prepare_tests()




# MAIN-FUNC: goes through all pre-preparation transformation steps of raw data
def prepare_and_save_raw_data():
    # load dataset from hugging-face
    # BUG: its only uploading 50
    dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset", split="train[:100]")  # define how many rows-example you want to load "train[:x]"

    # create outputc-column disease-classification-vector by feeding findings+impresssion cols of the raw data into openAI to classify each example, map() means apply the given function to every example in dataset
    dataset = dataset.map(generate_disease_vector)

    # create input-column patient-details synthetically by using rule-based function
    dataset = dataset.map(create_patient_details_column) 

    # create report output column by just combining findings+impression columns, and removing them
    dataset = dataset.map(combine_findings_and_impression_cols, remove_columns=["findings", "impression"]) 

    # go through each example and upload its image to s2-bucket and create new image-url column for it, and remove old image
    dataset = dataset.map(upload_image_and_create_image_url_column_single_example, remove_columns=["image"])

    # upload the raw-data as a parquet file to s3, with image-url column and input-output columns
    save_dataset_as_parquet_in_s3(dataset)  

    print("DONE...Uploading Raw Data to S3.")   # view uploaded data in a online parquet viwer just download it from s3

prepare_and_save_raw_data()









"""
Sample Output of Tests without Running:

-----CREATE DISEASE CLASSIFICATION OUTPUT COLUMN-----:
Example #1: {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x129F8FD90>, 'findings': 'The lungs are clear of focal consolidation, pleural effusion or pneumothorax. The heart size is normal. The mediastinal contours are normal. Multiple surgical clips project over the left breast, and old left rib fractures are noted. ', 'impression': 'No acute cardiopulmonary process.', 'disease_classification_vector': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

Example #2: {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x129F8FD90>, 'findings': 'Lung volumes remain low. There are innumerable bilateral scattered small pulmonary nodules which are better demonstrated on recent CT. Mild pulmonary vascular congestion is stable. The cardiomediastinal silhouette and hilar contours are unchanged. Small pleural effusion in the right middle fissure is new. There is no new focal opacity to suggest pneumonia. There is no pneumothorax. ', 'impression': 'Low lung volumes and mild pulmonary vascular congestion is unchanged. New small right fissural pleural effusion. No new focal opacities to suggest pneumonia.', 'disease_classification_vector': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]}

Example #3: {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x12A21A9E0>, 'findings': 'Lung volumes are low. This results in crowding of the bronchovascular structures. There may be mild pulmonary vascular congestion. The heart size is borderline enlarged. The mediastinal and hilar contours are relatively unremarkable. Innumerable nodules are demonstrated in both lungs, more pronounced in the left upper and lower lung fields compatible with metastatic disease. No new focal consolidation, pleural effusion or pneumothorax is seen, with chronic elevation of right hemidiaphragm again seen. The patient is status post right lower lobectomy. Rib deformities within the right hemithorax is compatible with prior postsurgical changes. ', 'impression': 'Innumerable pulmonary metastases. Possible mild pulmonary vascular congestion. Low lung volumes.', 'disease_classification_vector': [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]}
Example #1: {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x12A21AB10>, 'findings': 'The lungs are clear of focal consolidation, pleural effusion or pneumothorax. The heart size is normal. The mediastinal contours are normal. Multiple surgical clips project over the left breast, and old left rib fractures are noted. ', 'impression': 'No acute cardiopulmonary process.', 'disease_classification_vector': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'patient_details': '72 year old male PA view , tenderness, pain with deep breathing'}

Example #2: {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x12A21AB10>, 'findings': 'Lung volumes remain low. There are innumerable bilateral scattered small pulmonary nodules which are better demonstrated on recent CT. Mild pulmonary vascular congestion is stable. The cardiomediastinal silhouette and hilar contours are unchanged. Small pleural effusion in the right middle fissure is new. There is no new focal opacity to suggest pneumonia. There is no pneumothorax. ', 'impression': 'Low lung volumes and mild pulmonary vascular congestion is unchanged. New small right fissural pleural effusion. No new focal opacities to suggest pneumonia.', 'disease_classification_vector': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0], 'patient_details': '70 year old female PA view , weight loss, nighttime breathlessness'}

Example #3: {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x12A21AB10>, 'findings': 'Lung volumes are low. This results in crowding of the bronchovascular structures. There may be mild pulmonary vascular congestion. The heart size is borderline enlarged. The mediastinal and hilar contours are relatively unremarkable. Innumerable nodules are demonstrated in both lungs, more pronounced in the left upper and lower lung fields compatible with metastatic disease. No new focal consolidation, pleural effusion or pneumothorax is seen, with chronic elevation of right hemidiaphragm again seen. The patient is status post right lower lobectomy. Rib deformities within the right hemithorax is compatible with prior postsurgical changes. ', 'impression': 'Innumerable pulmonary metastases. Possible mild pulmonary vascular congestion. Low lung volumes.', 'disease_classification_vector': [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 'patient_details': '55 year old female AP view , hypertension , leg swelling'}

-----CREATE REPORT OUTPUT COLUMN-----:
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 430.97 examples/s]
Example #1: {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x12A21A9E0>, 'disease_classification_vector': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'patient_details': '72 year old male PA view , tenderness, pain with deep breathing', 'report': 'The lungs are clear of focal consolidation, pleural effusion or pneumothorax. The heart size is normal. The mediastinal contours are normal. Multiple surgical clips project over the left breast, and old left rib fractures are noted. No acute cardiopulmonary process.'}

"""