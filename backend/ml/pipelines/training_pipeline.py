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
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # hf tokenizers can deadlock with forked workers on macOS
from transformers.utils import logging as hf_logging    # quit the HF logs because they are spammy
hf_logging.set_verbosity_error()  
import torch
import torch.nn as nn            # neural network layers modules
import torchvision.models as tv  # ready-made CNN backbones (ResNet, etc
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader    # this is how we do stack the batched tensors in training-inference pipe
from torchvision.models import ResNet50_Weights     # backbone weights
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModel, AutoConfig      # for text-encoder: pip install transformers torch
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
TEXT_ENCODER_MODEl_NAME = "bert-base-uncased"
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
    

    # Builds an optimizer with sensible param groups for the current phase
    # phase=1, then only train heads (project + optional classifer)
    # phase=2, discriminative LRs: smaller for backbone, larger for heads
    # weight-decay: is for l2-style regularization, helps prevent overfitting
    # optimizer_cls:  is torch object, specifies which optimizer algo to build.
    def build_optimizer(self, phase, lr_backbone=1e-4, lr_head=5e-4, weight_decay=1e-2, optimizer_cls=torch.optim.AdamW):
        
        # if its phase 1 we only build an optimizer that will only update the heads
        if phase == 1:
            # params is a list of parameter groups, where the first group is the projection-head-params trained to with learning-rate lr-head, add this dict-group to params
            params = [{"params": self.proj.parameters(), "lr": lr_head}]
            # the second group is the classifier-head-params added to params-list
            if self.classifier is not None:
                params.append({"params":self.classifier.parameters(), "lr":lr_head})

            # create and return the torch-optimizer using those parameters groups for the 2 heads
            # torch optimizer only trains parameter groups we give it and since we didnt give it backbone-params it wont train backbone
            return optimizer_cls(params, weight_decay=weight_decay)

        # if its phase 2 (fine-tuning-mode) then we build an optimizer that will update heads + backbone
        elif phase == 2:
            params = []     # start with empty list of parameter groups

            # for every parmeter in backbone-model if it requires-grad=true then only add it to bb-params (supports partial unfreeze)
            bb_params = [p for p in self.backbone.parameters() if p.requires_grad]
            # if backbone-params has anything in it then add it as a parameter group to params-list
            if bb_params:
                params.append({"params": bb_params, "lr":lr_backbone})

            # always train the projection-head so add its params as a parameter group
            params.append({"params":self.proj.parameters(), "lr":lr_head}) 
            # always train the classifier-head so add itts params as a parameter group to params list
            if self.classifier is not None:
                params.append({"params": self.classifier.parameters(), "lr": lr_head})

            # create and return torch-optimizer using parameter-groups for the 2 heads, and parameter-group for backbone
            return optimizer_cls(params, weight_decay=weight_decay)


    # ---------- Forward passes ----------, underscore means private method not used outside class

    # used in phase 1 to save memory/compute while frozen: decorator dsiables autograd for everything inside the function, autograd=differentiation engine, in phase 1 the backbone is frozen we don't want to build a computation graph for it so it speeds up forward
    @torch.no_grad()    
    def _backbone_forward_nograd(self, x):  # takes in torch-tensor input, batch of input images shape [B, 3, H, W]
        # runs the pretrained-cnn (resnet without final) on input images x.
        # for resnet the last layer is a global average pool so shape is [B, feat-dim, 1, 1] = [B, 2048, 1, 1]
        feats = self.backbone(x)       
        # collapses dimensions form index 1 onward turning [B, feat_dim, 1, 1] into flat feature vector [B, feat-dim]
        # this is what project head explains
        return feats.flatten(1)     
    
    # used in phase 2 normal training with gradients, dont disable autograd
    def _backbone_forward_grad(self, x):
        feats = self.backbone(x)    # feeds batch of input-images into backbone-model, [B, feat_dim, 1, 1]
        return feats.flatten(1)     # [B, feat_dim], same as above not disabling autograd
    
    # returns image embeddings only (no classifer) the output of our image-encoder-cnn
    # images: is input torch-tensor of shape [B, 3, H, W], H=height, W=weight of pixel
    def encode(self, images):
        # if backbone is frozen then do no-grad forward-pass for backbone-model passing images-input
        if self.is_backbone_frozen == True:
            feats = self._backbone_forward_nograd(images)
        # if backbone is not frozen then do grad forward-pass for backbone-model passing images-input
        else:
            feats = self._backbone_forward_grad(images) # in both cases feats

        # passes backbone features (which are its output) into the projection-head which above is meant to convert it to the d_img embedding size we want. 
        # [B, d_img], d-img=1024, this is the final embedding shape of all images batched up, for each image outputs a tensor that represents it that its it embedding image
        z = self.proj(feats)
        return z
    
    # standard forward for warm-up training.
    # returns dict with embeddings and if enabled logits for disease labels
    def forward(self, images):
        z = self.encode(images)         # call encode func which returns mebeddings for every image [B, d_img]
        out = {"embeddings": z}         # dict with ebeding key and z value
        if self.classifier is not None:
            out["logits"] = self.classifier(z)   # [B, n_disease]
        return out




# ===========================================================
# Text Encoder
# ===========================================================

# tokenize input means to convert raw text into sequence on number so transformer can understand. 
# create tokenizer-class from transformers-torch library based on the model-name provided.
# from-pretrained loads the tokenizer files from HF hub for bert so the text is split into same token-ids that the model expects
tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_MODEl_NAME)

# takes in list of strings, where each string is a patient-detail, of length B the batch-size, pad every sequence
# returns dict of tensors suitable for model
def tokenize_patient_details(text_list, max_len=96):

    # call the tokenizer we created, passing the list of texts, if a string would exceed max-len then cut it off so shapes of each string are [B, seq-len], each token is a sub-word so 96 tokens is approzimately 65 worxs
    tok = tokenizer(text_list, padding="max_length", truncation=True, return_tensors="pt", max_length=max_len)

    # dictionary of tensors, key "input-ids" is equal to tensor of shape [B, L] each tensor in batch is represents a patient-detial-example-text of length L in text-list
    # key "attention_mask" is eqault ot torch tensor of shape [B, L]
    return tok

# torch-model that represents an text-encoder-transformer, takes in as input the batch of patient-detail texts
# backbone: is self.encoder, its a bert-model backbone is suaully called the encoder.
# the idea is as same as the cnn, where we have a backbone pre-trained-transformer and we add head(s) on stop of it. 
# heads: pooling-head, projection-head (just converts it into the dimentionality we want), classifier-head
class TextEncoderTransformer(nn.Module):

    def __init__(self, model_name="bert-base-uncased", d_txt=512, n_disease=13, use_warmup_classifier=True):
        super().__init__()
        self.model_name = model_name
        self.d_txt = d_txt              # embedding size of a patient-detail string
        self.n_disease = n_disease      # number of disease classes
        self.use_warmup_classifier=use_warmup_classifier      # if you want a classifer-head

        # loads the given models hyperparameters from torch
        self.hyperparameters = AutoConfig.from_pretrained(model_name)
        # loads the pretrained-transfoer-encoder-backbone-model from torch given its name
        self.encoder = AutoModel.from_pretrained(model_name)

        # stores the encoders token-embedding dimensionality so we can wire the layers correctly even if you swap models
        self.hidden_size = self.hyperparameters.hidden_size
        # create the projection-head maps the encoders pooled vector from hidden-size to our desired embedding size d-txt
        self.proj = nn.Linear(self.hidden_size, self.d_txt)
        # optional classifier-head linear-transformation layer, for disease supervision during phase-1, takes the 
        self.classifier = nn.Linear(d_txt, n_disease) if self.use_warmup_classifier else None    # takes projected embedding d-txt -> logits [n-disease]

        self.is_frozen = False  # internal flag if backbone-pretrained-model is frozen or not

    # PHASE 1: freeze the bacbone-encoder, only train heads
    def freeze_encoder(self):
        # iterate all parameter-tensors of backbone-encoder-model & turn off its gradient updates so optimizer wont change these weights, torch wont allocate grad buffers for them
        for p in self.encoder.parameters():
            p.requires_grad = False
        # update flag
        self.is_frozen = True  

        # puts the encoder in evaluation-mode, for bert-family models this disables dropout giving stable outputs when frozen
        self.encoder.eval()

        # keep the projection-head in training-mode, this head maps hidden_size → d_tx, this doesnt itself train
        self.proj.train()

        # keep classifier-head in training-mode, multi-label disease logits
        if self.classifier is not None:
            self.classifier.train()
    
    # PHASE 2: allows the pretrained backbone transformer to learn again, unfreeze backbone keep training heads
    def unfreeze_encoder(self):
        # iterate all parameter-tensors in backbone-encoder-model, turn on gradients so backprop can update them during fine-tuing
        for p in self.encoder.parameters():
            p.requires_grad = True
        # update flag
        self.is_frozen = False

        # puts the encoder-backboe is training mode, enables dropout.
        self.encoder.train()

        # puts projection-head in training mode, it continues to learn algonside the encoder
        self.proj.train()

        # puts classifier-head in training mode
        if self.classifier is not None:
            self.classifier.train()

    # builds an optimizer, specifying learning-rates for encoder & heads, optimizer_cls: is torch object, specifies which optimizer algo to build, phase=1/2
    def build_optimizer(self, phase, lr_enc=1e-4, lr_head=54-4, weight_decay=1e-2, optimizer_cls=torch.optim.AdamW):
        # if its phase 1 only train heads so pass the parameters groups for heads only not backbone
        if phase == 1:
            # params is a list of parameter groups, the first group is the project-head parameters add its group in params-list, passing in the learning-rate for the head in this param group
            params = [{"params":self.proj.parameters(), "lr": lr_head}]
            # add classifer param group {} to params, passing in the classifier-parameters
            if self.classifier is not None:
                params.append({"params": self.classifier.parameters(), "lr": lr_head})
            return optimizer_cls(params, weight_decay=weight_decay)
        
        # if its phase 2 add parameter groups for backbone & head
        if phase == 2:  
            # gather only the encoder-backbone parameters that are currently trainable ie param.requires-grad=true
            enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
            params = []

            # if any exist add a parameter group for the encoder's-params
            if enc_params:
                params.append({"params":enc_params, "lr":lr_enc})

            # create parameter-groups for projection-head and classifier-head
            params.append({"params": self.proj.parameters(), "lr": lr_head})
            if self.classifier is not None:
                params.append({"params": self.classifier.parameters(), "lr": lr_head})
            return optimizer_cls(params, weight_decay=weight_decay)

    """
    Given a patient detials: "67M smoker"
    Tokenize -> split into tokens -> token1=67, token2=M, ....etc.
    Each tokens becomes an integer shape [L], where L is seq-length

    Transformer forward -> for eaech token the model produces a vector of size H (hidden size 768), stack those per token last-hidden-state shape [L,H]
    [token-0-vector, token-1-vector, ... , token-L-1-vector]
    Token vector = is vector representing that token, the contextual embedding for a specific token position output by the model

    If you have a batch of B examples you get: last-hidden-state shape [B, L, H], so each example has its own [L, H] matrix.

    Each example (to feed the fusion) we want a single fixed-size embedding. 
    Masked mean pooling: average only the non-padding token vectors, result shape is [B, H], where one H-dim vector per example.
    Then we pass through projection head H->D_txt, z_txt shape [B, D_txt]
    """
    # masked-mean-pooling over token embeddings, it turns a sequence of token vectors into an vector per example by averaging only the real tokens (not-padded), a sequence/example has multiiple tokens so multiple token-vectors
    # last-hidden-state: the transformers final-layer-output token embeddings, shape [B, L, H] batch, seq-len, hidden-size. Its a torch tensor. Each row is a toekn vector
    # attention-mask: tells which positions real tokens (1) and which are padding (0), shape [B, L]. Its a torch tensor. Comes from the tokenizer.
    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,L,1]
        # element-wise multiply zeros out padded token vectors (because pad mask positions are 0).
        summed = (last_hidden_state * mask).sum(dim=1)                   # [B,H]
        # counts how many tokens were valid per example (since mask has 1s for valid tokens). Shape [B, 1]
        counts = mask.sum(dim=1).clamp(min=1e-6)                         # [B,1]
        # divides feature-wise totals by the number of valid tokens → the mean embedding per example. Shape [B, H]
        return summed / counts

    # Forward functions below are encode(), forward()
    # input-ids: [B, L], token ids is the input the text encoder needs, what you get from the tokenizer, where L=seq-len and each example in batch represents a patient-detail example
    # attention-mask: [B, L] 1 for real tokens, 0 for padding so we can ignore pads
    # token-type-ids: [B, L] segment ids
    def encode(self, input_ids, attention_mask, token_type_ids):
        # if backbone is frozen run teh forward of the transformer without gradient
        if self.is_frozen == True:
            with torch.no_grad():
                # forward-pass of pretrained-backbone-transformer-bert it returns hidden states, output is an object with named attributes
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        else:  
            # if the encoder-backbone is not frozen phase-2 fine-tuning, run the forward pass with gradients so backprop can update the encoder weights
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)

        # out.last_hidden_state is sequence of toekn vectors from the last transformer layer [B, L, H], each [L, H] row for an example is the contextual embedding for each token position, pooling sits on top of final-transformer-layer
        # does masked pooling over the token dimension using attention-mask so pading doesnt affect the average
        # result is one vector per example, shape [B, H]
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)

        # project-head is linear layer that maps the pooled vector from H into our desired text-embedding size d-txt, ex 768->512
        # output z is the final text embedding per example [B, d_txt]
        z = self.proj(pooled)

        """
        In:
        input_ids [B,L], attention_mask [B,L] (and maybe token_type_ids [B,L])

        Encoder out:
        last_hidden_state [B,L,H]

        Pool:
        pooled [B,H]

        Project:
        z [B,d_txt] ← this is what you concatenate with the image embedding later.
        """

        return z
    

    # expects keys inputs_ids [B, L], attention_mask [B,L], (optional) token_type_ids
    # returns  {"embeddings": [B,d_txt], "logits": [B,n_disease]}
    def forward(self, **batch):
        z = self.encode(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
        out = {"embeddings": z}
        if self.classifier is not None:
            out["logits"] = self.classifier(z)
        return out

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
    # pil_img = Image.open(io.BytesIO(image_bytes_obj))       # convert image-bytes into pil-img-obj
    # tensor_img = image_transfom_into_tensor(pil_img)                    # convert pil-img into tensor format to be fed into cnn image encoder
    # print(tensor_img.shape)  # should be [3, 224, 224], [3, img_sze, img_sze], check image-size constant
    # print(tensor_img.dtype) 

    print("----------IMAGE ENCODER: CONSTRUCT IMAGE-CXR-TORCH-DATASET----------")
    # img_s3_key_inputs, disease_classification_vectors_labels = construct_input_label_pairs_for_image_encoder_dataset(features_labels_df)    # pass in df we loaded from feature-store
    # assert(len(img_s3_key_inputs) == len(disease_classification_vectors_labels))
    # dataset = CXR_ImageDataset(img_s3_keys_input=img_s3_key_inputs, bucket=os.getenv("AWS_S3_BUCKET_NAME"), labels=disease_classification_vectors_labels, image_transform=image_transfom_into_tensor)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True )   # create dataloader object
    # print(f"dataset length: {len(dataset)}")

    # for batch_imgs, batch_labels in dataloader:     # iterate for all batches in dataloader, it calls __getitem__ & __len__ in the abckground while doing this to fetch examples by index
    #     print(f"batch img input shape: {batch_imgs.shape}")  # [32, 3, 224, 224], 32=batch-sizes, [B, 3, H, W]
    #     print(f"batch disease-vec label shape: {batch_labels.shape}")  # [32, 14], [batch, disease-classification]
    #     print(f"img one example shape: {batch_imgs[0].shape}")          # [3, 244, 244]
    #     print(f"disease-vec one example value: {batch_labels[0]}")
    #     break

    print("---------IMAGE ENCODER: CREATE IMAGE-ENCODER-CLASS----------")
    # device = torch.device("cpu")
    # model = ImageEncoderCNN(backbone_name="resnet50", d_img=1024, n_disease_classes=13).to(device)
    # criterion = BCEWithLogitsLoss() 

    # Phase 1: freeze backbone, train heads
    print("=====Phase #1=====")
    # model.freeze_backbone()
    # optim = model.build_optimizer(phase=1, lr_head=5e-4, weight_decay=1e-2)
    # # iterate every batch of images/labels in dataloader
    # # where imgs: [B, 3, 224, 224], [32, 3, 224, 224], each element in batch is tensor representing image
    # # where y: [B, 13], [32, 13]
    # for imgs, y in dataloader:                    # passing in our data which fine-tunes it
    #     imgs = imgs.to(device); y = y.to(device)  # move both tensors to same compute device
    #     optim.zero_grad()       # clears old gradients stored in optimizer from previous step (otherwise they accumulate)
    #     out = model(imgs)       # runs forward pass on entire image-encoder pass in cur-imgs batch, out["embeddings"] [B, d_img] where each element is embedding-vector for image
    #     logits = out["logits"]  # pulls classification logits [B, 13]
    #     loss = criterion(logits, y) # computes loss given logits & labels
    #     loss.backward()             # backpropagates computes dloss/dtheta for all traiable params that participated in forward pass, heads only in Phase-1; heads + unfrozen backbone in Phase-2
    #     optim.step()                # updates params using optimizer with current gradients

    #     print("embeddings:", out["embeddings"].shape) # expect [B, 1024], [B, d_img]
    #     print("logits:",     out["logits"].shape)     # expect [B, 13], [B, disease-classes]
    #     break

    # Phase 2: unfreeze + discriminative lrs
    print("=====Phase #2=====")
    # model.unfreeze_backbone()
    # optim = model.build_optimizer(phase=2, lr_backbone=1e-4, lr_head=5e-4, weight_decay=1e-2)
    # for imgs, y in dataloader:                    # passing in our data which fine-tunes it
    #     imgs = imgs.to(device); y = y.to(device)  # move both tensors to same compute device
    #     optim.zero_grad()       # clears old gradients stored in optimizer from previous step (otherwise they accumulate)
    #     out = model(imgs)       # runs forward pass on entire image-encoder pass in cur-imgs batch, out["embeddings"] [B, d_img] where each element is embedding-vector for image
    #     logits = out["logits"]  # pulls classification logits [B, 13]
    #     loss = criterion(logits, y) # computes loss given logits & labels
    #     loss.backward()             # backpropagates computes dloss/dtheta for all traiable params that participated in forward pass, heads only in Phase-1; heads + unfrozen backbone in Phase-2
    #     optim.step()                # updates params using optimizer with current gradientss

    #     print("embeddings:", out["embeddings"].shape) # expect [B, 1024], [B, d_img], each element in out["eembeddings"] is a embedding-vector representing that im
    #     print("logits:",     out["logits"].shape)     # expect [B, 13], [B, disease-classes]
    #     break
    



    print("----------TEXT ENCODER: TOKENIZE PATIENT DETAILS TEXT----------")
    device = torch.device("cpu")
    # create synthetic batch
    patient_details_texts = ["67M, smoker; dyspnea; CHF history.", "54F, no smoking; cough; asthma."]
    n_classes = 13
    # create torch-tensor of labels for each example of patient-details
    y = torch.tensor([[0,1,0,1,0,0,0,0,1,0,1,0,0], [1,0,0,0,0,1,0,0,0,0,0,0,1]], dtype=torch.float32, device=device) 
    
    # tokenize the texts into tensors & build model
    tok = tokenize_patient_details(patient_details_texts, max_len=96)   # length of each sequence is set to 96
    text_encoder_model = TextEncoderTransformer(model_name="bert-base-uncased", d_txt=512, n_disease=13).to(device)

    # stuff returned from tokenizer
    print(f"Batch tensors of patient details: {tok['input_ids'].shape} ")   # expected [B, seq-len] = [2, 96], we get text length above
    print(f"Single patient detail text tensor tokenized shape: {tok['input_ids'][0].shape}")  # expect shape [seq-len] = [96]
    
    # loss for warm-up classifier
    criterion = nn.BCEWithLogitsLoss()
   

    print("=====Phase #1, freeze encoder, train heads======")
    text_encoder_model.freeze_encoder()         # freeze-backbone-model
    optim = text_encoder_model.build_optimizer(phase=1, lr_head=5e-4)   # create

    text_encoder_model.train()  # method inheritied from torch

    # run 3 update-steps for testing
    print("Training...")
    for step in range(3):
        optim.zero_grad()                   # clears old gradients
        out = text_encoder_model(**tok)     # forward-pass through TextEncoderTransformer obj, tok has the inputs look above
        z = out["embeddings"]               # pooled+projected-text-vectors [B, d_txt], for each example there is a vector embedding output from transformer of size d-txt
        logits = out["logits"]              # logits for diseases
        loss = criterion(logits, y)         # compute the loss pass ing multi-label targets y, which is disease-classification-vector for each patient-detail example in B
        loss.backward()                     # backprop autograd computes gradients for all parameters were used to produce logits and requires-grad=True
        optim.step()                        # update stpe uses accumlated gradietns to adjust parameters   
        # expected: z.shape = [B, d_txt] = [2, 512], loss going down                 
        print(f"Step {step}: z.shape={z.shape}, logits.shape={logits.shape}, loss={loss.item():.4f}")


    print("=====Phase #2=====")
    text_encoder_model.unfreeze_encoder()   # unfreeze the backbone make its params trainable
    optim = text_encoder_model.build_optimizer(phase=2, lr_enc=2e-5, lr_head=5e-4, weight_decay=1e-2)
    
    print("Training...")
    for step in range(3):
        optim.zero_grad()                   # clears old gradients
        out = text_encoder_model(**tok)     # forward-pass through TextEncoderTransformer obj, tok has the inputs look above
        z = out["embeddings"]               # pooled+projected-text-vectors [B, d_txt], for each example there is a vector embedding output from transformer of size d-txt
        logits = out["logits"]              # logits for diseases
        loss = criterion(logits, y)         # compute the loss pass ing multi-label targets y, which is disease-classification-vector for each patient-detail example in B
        loss.backward()                     # backprop autograd computes gradients for all parameters were used to produce logits and requires-grad=True
        optim.step()                        # update stpe uses accumlated gradietns to adjust parameters   
        # expected: z.shape = [B, d_txt] = [2, 512], loss going down                 
        print(f"Step {step}: z.shape={z.shape}, logits.shape={logits.shape}, loss={loss.item():.4f}")


training_tests()




