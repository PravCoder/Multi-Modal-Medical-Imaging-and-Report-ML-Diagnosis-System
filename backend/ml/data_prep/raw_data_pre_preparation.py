# File: prepares the raw data before it is transformed in the feature pipeline. This is the "pre-preparation" since we do not have a single dataset for our needs. 
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
import json
import random
from helper import DISEASES, SYMPTOMS_MAP
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




def main():
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


    # print("\n-----CREATE PATIENT DETAILS INPUT COLUMN-----:")
    # we are taking the dataset and creating the input column patient-detials using arule-based function
    small_dataset = small_dataset.map(create_patient_details_column)    # make sure this is same dataset as above with disease-classification column else it wont work, apply func to all examples in dataset
    print(f"Example #1: {small_dataset[0]}")
    print(f"\nExample #2: {small_dataset[1]}")
    print(f"\nExample #3: {small_dataset[2]}")


    print("\n-----CREATE REPORT OUTPUT COLUMN")
    # we are taking the dataset and creating the output column report
    small_dataset = small_dataset.map(combine_findings_and_impression_cols, remove_columns=["findings", "impression"])  # apply func all examples in daataset and remove cols
    print(f"Example #1: {small_dataset[0]}")        # print a whole row






main()