# File: prepares the raw data before it is transformed in the feature pipeline. This is the pre-preparation. 
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt

dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset", split="train")

print(f"\nFeatures: {dataset.features}")
print(f"Number of Examples: {len(dataset)} rows.")

# First row sample
sample = dataset[0] 
print(f"\nSample Row: {sample}")  # dict with keys the rows names "image", "findings", "impressions"
print(f"Image Shape: {sample["image"].size}")  # (width, height)

# Iterate all images and make sure they can be opened
for idx, row in enumerate(dataset):
    try:
        img = row["image"]  # already a PIL.Image object
        img.verify()        # verify image integrity
    except Exception as e:
         print(f"Error with image at index {idx}: {e}") # if no expection then all rows images can be opened

# View Image of an example, PIL.Image object
img = sample["image"]   
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title(f"Impression: {sample['impression']}\nFindings: {sample['findings']}")
plt.show()