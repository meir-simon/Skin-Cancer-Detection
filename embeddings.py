import pandas as pd
import os.path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as v2
import numpy as np
from tqdm import tqdm
from transformers import ViTModel, ViTFeatureExtractor
import csv
import json

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print device information
if device.type == 'cuda':
    print("Using GPU for computation.")
else:
    print("Using CPU for computation.")

IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IN_COLAB = "COLAB_GPU" in os.environ

if IN_COLAB:
    # This code block is for Colab environment
    # from google.colab import userdata
    # # only the first time
    # os.environ["KAGGLE_KEY"] = "705abdae60770428ed9d890d9d96dff3"
    # os.environ["KAGGLE_USERNAME"] = "shukitornheim"

    # ! kaggle competitions download isic-2024-challenge
    # ! unzip isic-2024-challenge.zip
    IMG_DIR = "/content/train-image/image"
    CSV_PATH = "/content/train-metadata.csv"
elif IN_KAGGLE:
    IMG_DIR = "/kaggle/input/isic-2024-challenge/train-image/image"
    CSV_PATH = "/kaggle/input/isic-2024-challenge/train-metadata.csv"
else:
    IMG_DIR = "/Users/yuda/Desktop/data_bases/isic-2024-challenge/train-image/image"
    CSV_PATH = "/Users/yuda/PycharmProjects/my_isic2024/isic-2024-challenge/train-metadata.csv"

IMG_SIZE = (50, 50)
RANDOM_STATE = 40

ISIC_df = pd.read_csv(CSV_PATH)
ISIC_df = ISIC_df[["isic_id"]]
ISIC_df["img_path"] = ISIC_df["isic_id"].apply(lambda id: f"{os.path.join(IMG_DIR, id)}.jpg")
ISIC_df.drop(["isic_id"], axis=1, inplace=True)

transform = v2.Compose([
    v2.Resize(IMG_SIZE),
    v2.ToTensor(),
    v2.Normalize(mean=[0.6298, 0.5126, 0.4097], std=[0.1386, 0.1308, 0.1202]),
])

def load_and_transform_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

class EmbeddingExtractor(nn.Module):
    def __init__(self, model_name='resnet18'):
        super(EmbeddingExtractor, self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.features.children()))
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        elif model_name == 'efficientnet_v2_m':
            self.model = models.efficientnet_v2_m(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        elif model_name == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        elif model_name == 'vit_b_16':
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        else:
            raise ValueError("Model name not recognized.")
        
        self.to(device)  # Move the model to the appropriate device (GPU or CPU)

    def forward(self, x):
        if hasattr(self, 'feature_extractor'):
            outputs = self.model(x).last_hidden_state
            x = torch.mean(outputs, dim=1)
        else:
            x = self.model(x)
            x = torch.flatten(x, start_dim=1)
        return x

    def process_images(self, image_list):
        self.eval()
        embeddings_list = []
        
        with torch.no_grad():
            for image_tensor in image_list:
                if isinstance(image_tensor, torch.Tensor):
                    image_tensor = image_tensor.to(device)  # Move the tensor to the appropriate device
                    if hasattr(self, 'feature_extractor'):
                        image_tensor = image_tensor.permute(1, 2, 0)  # CHW to HWC
                        image = Image.fromarray((image_tensor.numpy() * 255).astype(np.uint8))
                        inputs = self.feature_extractor(images=image, return_tensors="pt")
                        image_tensor = inputs["pixel_values"].squeeze(0).to(device)  # Move tensor to device
                    embedding = self.forward(image_tensor.unsqueeze(0))
                    embedding_np = embedding.squeeze().cpu().numpy()  # Move tensor back to CPU before converting to numpy
                    embeddings_list.append(embedding_np)
                else:
                    raise ValueError("Each image tensor must be a torch.Tensor")
        
        return embeddings_list

def load_images_in_batches(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [load_and_transform_image(path) for path in batch_paths]
        yield batch_paths, batch_images


def create_embeddings_json(model_name, embeddings_list, batch_size=1000):
    row_len = len(embeddings_list[0])
    headers = [f"{model_name}_embedding{i}" for i in range(1, row_len + 1)]
    file_name = f"{model_name}_embeddings.json"
    
    with open(file_name, 'w') as file:
        # Write headers as the first line
        json.dump([headers], file)
        file.write('\n')  # Adding a newline to separate headers from the data
        
        # Write embeddings in batches with tqdm for progress tracking
        for i in tqdm(range(0, len(embeddings_list), batch_size), desc="Writing embeddings"):
            batch_embeddings = embeddings_list[i:i + batch_size]
            # Convert numpy arrays to lists
            batch_embeddings_list = [embedding.tolist() for embedding in batch_embeddings]
            json.dump(batch_embeddings_list, file)
            file.write('\n')  # Separate batches with a newline

    return file_name



batch_size = 16
image_paths = ISIC_df["img_path"].tolist()

# models = ['resnet18','resnet50','vgg16', 'efficientnet_b0', 'efficientnet_v2_m', 'mobilenet_v3_small', "vit_b_16"]
model_name = 'vgg16'
extractor = EmbeddingExtractor(model_name=model_name)

embeddings_list = []
for batch_paths, batch_images in tqdm(load_images_in_batches(image_paths, batch_size), total=len(image_paths) // batch_size, desc='Processing Images'):
    image_tensors = torch.stack(batch_images).to(device)  # Move image tensors to the appropriate device
    embeddings_batch = extractor.process_images(image_tensors)
    embeddings_list.extend(embeddings_batch)

file_name = create_embeddings_json(model_name, embeddings_list)
from google.colab import files
files.download(file_name)


#def read_embeddings_json(file_name):
    # embeddings_list = []
    # with open(file_name, 'r') as file:
    #     headers = json.load(file)  # קריאת הכותרות
    #     for line in file:
    #         embeddings_list.extend(json.loads(line))  # קריאת והוספת קבוצות
    # return headers, embeddings_list
