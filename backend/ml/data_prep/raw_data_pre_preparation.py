# File: prepares the raw data before it is transformed in the feature pipeline. This is the pre-preparation. 
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt


dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset", split="train")

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



# For each exmaple passes the findings + impression column text into openAI and asks it to find the diagnosis, this will be the disease diagnosis column.
def add_disease_diagnosis_column(sample):
    pass




def main():
    dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset", split="train")
    sample = dataset[0] 

    print("\nBASIC INFO + VALIDATE IMAGES")
    # show_basic_info(dataset)

    print("\nShOW IMAGE")
    # show_example_image(sample)

    print("ADD DISEASE DIAGNOSIS COLUMN")
    add_disease_diagnosis_column(sample)




main()