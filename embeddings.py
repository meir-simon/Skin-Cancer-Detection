import torch
import torchvision.models as models
import torch.nn as nn
import pandas as pd

class EmbeddingExtractor(nn.Module):
    """
    A class to extract embeddings from images using various pre-trained models.
    """

    def __init__(self, model_name='resnet18'):
        """
        Initialize the EmbeddingExtractor with the specified model.

        :param model_name: The name of the pre-trained model to use. Options include 'resnet18', 'resnet50', 
                           'vgg16', 'efficientnet_b0', 'efficientnet_v2_m', and 'mobilenet_v3_small'.
        """
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
        else:
            raise ValueError("Model name not recognized.")

    def forward(self, x):
        """
        Forward pass to extract features from the input images.

        :param x: Input tensor of images with shape (batch_size, channels, height, width).
        :return: Tensor containing the flattened feature vectors for each image.
        """
        x = self.model(x)
        x = torch.flatten(x, 1)  # Flatten the tensor to a 1D vector for each image
        return x

    def get_embeddings(self, images):
        """
        Get embeddings for a batch of images.

        :param images: Input tensor of images with shape (batch_size, channels, height, width).
        :return: Tensor containing the embeddings for each image in the batch.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            if isinstance(images, torch.Tensor):
                embeddings = self.forward(images)
            else:
                raise ValueError("Input must be a torch.Tensor")
        return embeddings

    def process_images(self, image_dict):
        """
        Process a dictionary of image IDs and tensors and return their embeddings.

        :param image_dict: Dictionary with image IDs as keys and image tensors as values.
        :return: DataFrame with image IDs and their corresponding embeddings.
        """
        self.eval()
        embeddings_list = []
        ids_list = []
        
        with torch.no_grad():
            for image_id, image_tensor in image_dict.items():
                if isinstance(image_tensor, torch.Tensor):
                    embedding = self.forward(image_tensor.unsqueeze(0))  # Add batch dimension
                    embeddings_list.append(embedding.squeeze().numpy())  # Remove batch dimension and convert to numpy
                    ids_list.append(image_id)
                else:
                    raise ValueError("Each image tensor must be a torch.Tensor")
        
        return pd.DataFrame({'image_id': ids_list, 'embedding': embeddings_list})

# Example usage
if __name__ == "__main__":
    model_name = 'resnet18'
    extractor = EmbeddingExtractor(model_name=model_name)
    
    # Create dummy images with image IDs
    image_dict = {
        'image1': torch.randn(3, 224, 224),
        'image2': torch.randn(3, 224, 224),
        'image3': torch.randn(3, 224, 224)
    }
    
    # Get embeddings
    embeddings_df = extractor.process_images(image_dict)
    print(embeddings_df)

    # print(embeddings_df)
