# TRAINING PIPELINE: 
import hopsworks
import pandas as pd
import numpy as np
import json, os
from helper import print_clean_df
import warnings
import io
from PIL import Image
import boto3
import torch
import torch.nn as nn            # neural network layers modules
import torchvision.models as tv  # ready-made CNN backbones (ResNet, etc
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader    # this is how we do stack the batched tensors in training-inference pipe
from torchvision.models import ResNet50_Weights     # backbone weights
from torch.nn import BCEWithLogitsLoss
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
    no_schema = url[5:]     # removes the s3://
    bucket, key = no_schema.split("/", 1)
    return bucket, key

# ===========================================================
# Image Encoder
# ===========================================================

# this is a torchvision transform-pipeline that accepts a PIL Image object
# it returns a torch.FloatTensor of shape [3, IMG_SIZE, IMG_SIZE] or [3, H, W] for three color channels because ImageNet is like that. 
# this is applied to each image individually and stacked the batch shape is [B, 3, IMG_SIZE, IMG_SIZE], [B, 3, H, W], where B is the batch size, 3 is color channels *** important
image_transfom_into_tensor = T.Compose([
    T.Resize(256, antialias=True),    # takes in pil-image, resizes so shorter side is 256-pixels
    T.CenterCrop(IMG_SIZE),           # crops the center square out of the image (224, 224, C)
    T.ToTensor(),                     # converts pil -> torch.FloatTensor, reorders dims to channel dim first [C, H, W], If grayscale input → [1, 224, 224]. If RGB input → [3, 224, 224], scale pixel values to [0,1]
    T.Lambda(lambda x: x.repeat(3,1,1) if x.size(0) == 1 else x),  # ensures 3 channels for colors, If input has C=1 (grayscale), repeats that channel 3 times: [1,224,224] → [3,224,224]. If already C=3, leaves unchanged.
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), # normalizes each channel seperately
    # output: [3, 224, 224] tensor representing image to be fed into cnn-image-encoder, 
])

# this is neccesary for the cxr-cnn-dataset-obj
def construct_input_label_pairs_for_image_encoder_dataset(df):
    parsed = [parse_s3_url(url) for url in df["image_url"].tolist()]     # get all image-url-strings in iamge-url col of df in order as list, parse the url correclty
    img_keys = [k for _, k in parsed]  

    y_labels = [np.asarray(v, dtype=np.float32) for v in df["disease_classification_vector"].tolist()]  # for every element in disease-classification-col add it to this list
    return img_keys, y_labels       # they are pairs because they are in order and hold same index in the two lists


# create a dataset object for all the images transformed and stacked to be fed into cnn encoder
class CXR_ImageDataset(Dataset):
    def __init__(self, img_s3_keys_input, bucket, labels=None, image_transform=None):
        self.img_s3_keys_input = img_s3_keys_input      # given the keys of all the images in s3-bucket stacked, this our input for this dataset
        self.bucket = bucket                # given s3-bucket name of where iamges are stored
        self.labels = labels                # given all the disease-classification-vectors for each examples tacked, this our labels for this dataset
        self.image_transform = image_transform      # given the pytorch-transform-pipeline-func that converts each image into the correct tensor form

    def __len__(self):  
        assert(len(self.img_s3_keys_input) == len(self.labels))            
        return len(self.img_s3_keys_input)  # number of examples in dataset

    def __getitem__(self, i):
        # get the image-bytes-object from s3-storage
        img_bytes = get_image_from_s3(self.bucket, self.img_s3_keys_input[i])   
        # convert image-bytes-object to pil-image 
        img = Image.open(io.BytesIO(img_bytes))     
        # use image-transform func we defined to convert pil-image-obj into tensor of shape [3, H, W] or [3, IMG_SIZE, IMG_SIZE], this is the input for example-i in this image-encoder-dataset
        x = self.image_transform(img)             
        # convert the disease-classification into torch-float-tensor for example-i
        y = torch.tensor(self.labels[i], dtype=torch.float32) 
        # return the (x,y) input-label pair for this example-i
        return x, y     
    

# torch module that represents a CNN-image-encoder that takes in a image & outputs a embedding that represents that image
# backbone of cnn: is itself a deep-cnn typically pre-trained on massive datasets, it process input data like images through multiple convolutional and pooling layers to generate rich, hierarchical feature maps.

class ImageEncoderCNN(nn.Module):
    
    def __init__(self, backbone_name="resnet50", d_img=1024, n_disease_classes=13, use_warmup_classifier=True, pretrained_weights=ResNet50_Weights.IMAGENET1K_V2):
        super().__init__()  # inherit from parent torch module

        self.backbone_name = backbone_name      # string name of backbone pre-trained cnn model
        self.d_img = d_img         # output embedding size, dimenstioanlity of the image embedding vector that comes out of the cnn image encoder
        self.n_disease_classes = n_disease_classes      # number of disease class we have
        # when true: You train projection → classifier with BCEWithLogitsLoss on the disease labels.
        # result: the projection learns to produce embeddings that already carry pathology signal before you unfreeze the CNN.
        self.use_warmup_classifier = use_warmup_classifier  

        self.pretrained_weights = ResNet50_Weights.IMAGENET1K_V2    # the weights of the backbone-cnn
        self.backbone = None   # is the pre-tranied-model's feature extractor with the final classification layer removed
        self.proj = None       # is the projection head thar maps the backbone-model's feature vector to the desired embedding size d_img
        self.classifier = None
        self.is_backbone_frozen = False
        self.load_pretrained_backbone()     # call this when object gets created
    
    def load_pretrained_backbone(self):
        if self.backbone_name.lower() == "resnet50":
            m = tv.resnet50(weights=self.pretrained_weights)            # create torch-res-net-model
            feat_dim = m.fc.in_features         # m.fc is the final fully-connected layer of res-net, .in_features tells you how many input features that layer expects. That number = size of pooled embedding coming out of the cnn its 2028 for resnet 50/101/152
            # list stuff retruns a list of the top-level modules inside the resnet-m, for res-net-50 the last element is m.fc
            # the sequential() wraps the remaining modules into a new sequential-layer
            # list(m.children()) = [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc], we removed the final classification layer
            self.backbone = nn.Sequential(*list(m.children())[:-1]) 
        else:
            raise ValueError(f"Unsupported backbone model: {self.backbone}")
        
        # project head: feat_dim -> d_img
        # create a fully connected linear layer that maps the resnet feature vector size to our desired embedding size 
        self.proj = nn.Linear(feat_dim, self.d_img)

        # Warm up classifier head for BCE loss
        if self.use_warmup_classifier == True:
            # create linear classifier head that sits on top of the image embedding, it maps embedding (B x d_img) to (B x n_disease_Classes)
            self.classifier = nn.Linear(self.d_img,self. n_disease_classes) 

        # track wheather backbone is frozen
        self.is_backbone_frozen = False

    # PHASE 1: freeze CNN weights and lock BN/dropotu stats. Train only new heads projection + optional classifer
    def freeze_backbone(self):
        # iterate every parameters tensors in pre-trained-backbone-cnn and tunrs off gradient tracking so no grads are computed for these tensors and the optimizer will not update them.
        for p in self.backbone.parameters():    # this is what is freezing the backbone 
            p.requires_grad = False
        self.is_backbone_frozen = True  # sets flag so other methods know backbone is frozen

        self.backbone.eval() # puts backbone is evaluation mode disables dropout and stops batchnorm from updating running mean.

        # ensures projection head is in training mode, heads still train duringi phase-1-freeze
        self.proj.train()

        # also train the classifer head put in training mode too so it learns during phase-1, train() does not train by itself it just puts it in that mode
        if self.classifier is not None:
            self.classifier.train()

    # PHASE 2: unfreeze CNN for fine-tuning. Use smaller lr for backbone
    def unfreeze_backbone(self):
        # iterate all paraemters tensors in backbone-model and turn gradient tracking back on, so they backbones weights can recieve gradients and be updated by the optimizer
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.is_backbone_frozen = False     # set so backbone is not frozen anymore

        # put pre-trained-backbone in training mode so batch-norm updates running mean.var using your data distribution, dropout is enabled is present
        self.backbone.train()

        # ensure projection-head is in training mode, this doesnt itself train
        self.proj.train()

        # keep warm-up classifier-head in training mode as well
        if self.classifier is not None:
            self.classifier.train()





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
    features_labels_df = load_features_labels_from_feature_store()
    print_clean_df(features_labels_df, num_rows=10)


    print("---------GET IMAGE FROM S3 AS BYTES OBJECT---------")
    s3_image_url_example = "s3://medical-ml-proj-bucket/chest-x-ray-images/cc4ff72b-a3adf0.jpg"
    bucket, key = parse_s3_url(s3_image_url_example)
    image_bytes_obj = get_image_from_s3(bucket, key)
    print(f"image bytes object: {image_bytes_obj[0:5]}")

    print("----------IMAGE ENCODER: IMAGE TRANSFORM TEST SINGLE IMAGE----------")
    pil_img = Image.open(io.BytesIO(image_bytes_obj))       # convert image-bytes into pil-img-obj
    tensor_img = image_transfom_into_tensor(pil_img)                    # convert pil-img into tensor format to be fed into cnn image encoder
    print(tensor_img.shape)  # should be [3, 224, 224], [3, img_sze, img_sze], check image-size constant
    print(tensor_img.dtype) 

    print("----------IMAGE ENCODER: CONSTRUCT IMAGE-CXR-TORCH-DATASET----------")
    img_s3_key_inputs, disease_classification_vectors_labels = construct_input_label_pairs_for_image_encoder_dataset(features_labels_df)    # pass in df we loaded from feature-store
    assert(len(img_s3_key_inputs) == len(disease_classification_vectors_labels))
    dataset = CXR_ImageDataset(img_s3_keys_input=img_s3_key_inputs, bucket=os.getenv("AWS_S3_BUCKET_NAME"), labels=disease_classification_vectors_labels, image_transform=image_transfom_into_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True )   # create dataloader object
    print(f"dataset length: {len(dataset)}")

    for batch_imgs, batch_labels in dataloader:     # iterate for all batches in dataloader, it calls __getitem__ & __len__ in the abckground while doing this to fetch examples by index
        print(f"batch img input shape: {batch_imgs.shape}")  # [32, 3, 224, 224], 32=batch-sizes, [B, 3, H, W]
        print(f"batch disease-vec label shape: {batch_labels.shape}")  # [32, 14], [batch, disease-classification]
        print(f"img one example shape: {batch_imgs[0].shape}")          # [3, 244, 244]
        print(f"disease-vec one example value: {batch_labels[0]}")
        break

    print("---------IMAGE ENCODER: CREATE IMAGE-ENCODER-CLASS----------")
    device = torch.device("cpu")
    model = ImageEncoderCNN(backbone_name="resnet50", d_img=1024, n_disease_classes=13).to(device)
    criterion = BCEWithLogitsLoss() 

    # Phase 1: freeze backbone, train heads
    print("=====Phase #1=====")
    model.freeze_backbone()

    # Phase 2: unfreeze + discriminative lrs
    print("=====Phase #2=====")
    model.unfreeze_backbone()



training_tests()