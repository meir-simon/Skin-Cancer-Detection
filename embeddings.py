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
    IMG_TEST_DIR = "insert"
    CSV_TEST_PATH = "INSERT"
else:
    IMG_DIR = "/home/mefathim/Desktop/Skin-Cancer-Detection-with-3D-TBP/train-image/image"
    CSV_PATH = '/home/mefathim/Desktop/Skin-Cancer-Detection-with-3D-TBP/train-metadata.csv'

IMG_SIZE = (50, 50)


ISIC_df = pd.read_csv(CSV_PATH)
ISIC_df = ISIC_df[["isic_id"]]
ISIC_df["img_path"] = ISIC_df["isic_id"].apply(lambda id: f"{os.path.join(IMG_DIR, id)}.jpg")
ISIC_df.drop(["isic_id"], axis=1, inplace=True)

if IN_KAGGLE:
    ISIC_TEST_df = pd.read_csv(CSV_TEST_PATH)
    ISIC_TEST_df = ISIC_df[["isic_id"]]
    ISIC_TEST_df["img_path"] = ISIC_df["isic_id"].apply(lambda id: f"{os.path.join(IMG_TEST_DIR, id)}.jpg")
    ISIC_TEST_df.drop(["isic_id"], axis=1, inplace=True)

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
        self.model_name = model_name
        weights_path = os.path.join('prepared_weights', f'{model_name}.pth') if not IN_KAGGLE else os.path.join('insert') 

        if model_name == 'resnet18':
            self.model = models.resnet18()
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        elif model_name == 'resnet50':
            self.model = models.resnet50()
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        elif model_name == 'vgg16':
            self.model = models.vgg16()
            self.model = torch.nn.Sequential(*list(self.model.features.children()))
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0()
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        elif model_name == 'efficientnet_v2_m':
            self.model = models.efficientnet_v2_m()
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        elif model_name == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small()
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        elif model_name == 'vit_b_16':
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        else:
            raise ValueError("Model name not recognized.")
        
        # Load the weights from the file
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path))
            print(f"Loaded weights for {model_name} from {weights_path}")
        else:
            raise FileNotFoundError(f"Weights file not found for {model_name} at {weights_path}")

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
    # headers = [f"{model_name}_embedding{i}" for i in range(1, row_len + 1)]
    file_name = f"{model_name}_embeddings.json"
    
    with open(file_name, 'w') as file:
        # Write headers as the first line
        # json.dump([headers], file)
        # file.write('\n')  # Adding a newline to separate headers from the data
        
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
image_test_paths = ISIC_TEST_df["img_path"].tolist()
embeddings_list = []
embeddings_test_list = []
path_and_list = [(ISIC_df,embeddings_list), (ISIC_TEST_df,embeddings_test_list)]
# models = ['resnet18','resnet50','vgg16', 'efficientnet_b0', 'efficientnet_v2_m', 'mobilenet_v3_small', "vit_b_16"]
model_name = 'vgg16'
extractor = EmbeddingExtractor(model_name=model_name)
# counter = 0
for path_and_list in path_and_list:
    image_paths, embeddings_list = path_and_list if IN_KAGGLE else image_paths, embeddings_list
    for batch_paths, batch_images in tqdm(load_images_in_batches(image_paths, batch_size), total=len(image_paths) // batch_size, desc='Processing Images'):
        image_tensors = torch.stack(batch_images).to(device)  # Move image tensors to the appropriate device
        embeddings_batch = extractor.process_images(image_tensors)
        embeddings_list.extend(embeddings_batch)
        # counter+= 1
        # if counter == 5:    
        #     break 


file_name = create_embeddings_json(model_name, embeddings_list)
# from google.colab import files
# files.download(file_name)


#def read_embeddings_json(file_name):
    # embeddings_list = []
    # with open(file_name, 'r') as file:
    #     headers = json.load(file)  # קריאת הכותרות
    #     for line in file:
    #         embeddings_list.extend(json.loads(line))  # קריאת והוספת קבוצות
    # return headers, embeddings_list
