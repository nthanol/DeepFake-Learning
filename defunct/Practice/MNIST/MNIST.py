import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms # Image Transforms and Augmentations
from torchvision import datasets

from torch.utils.data import Dataset

dtype = torch.float # TODO: Check time relevance
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU


# torchvision.datasets.MNIST('./',download = True) # Downloads the MNIST dataset from torchvision into the current directory

Transform = transforms.ToTensor() # Converts images into tensors

# Reads from MNIST directory and applies the corresponding transform to the data
train = datasets.MNIST(root='./', train = True, download = False, transform = Transform)
test = datasets.MNIST(root='./', train = False, download = False, transform = Transform)

class MNISTDataset(Dataset):
    """
    MAP-style. __getitem__ and __len__ methods need to be implemented.
    """
    def __init__(self, path, train): # train is a boolean
        Transform = transforms.ToTensor()
        data = datasets.MNIST(root=path, train = train, download = False, transform = Transform)
        self.images = [None] * len(data) #Holds the image pixel array
        self.labels = [None] * len(data) #Holds the number label
        for i in range(len(data)):
            self.images[i] = data[i][0]
            self.labels[i] = data[i][1]

    def __getitem__(self, index):
        x = self.images[index].float()
        x = torch.Tensor(x)
        x = torch.flatten(x) # flatten the image from a 28 x 28 to a 784 1-dimensional tensor
        return x

    # returns the length of the dataset
    def __len__(self):
        return len(self.labels)

class AutoEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim = 128):
        super().__init__()

        # 2 layers for the encoder
        self.encoder_l1 = nn.Linear(in_features=input_shape, out_features=latent_dim)
        self.encoder_l2 = nn.Linear(in_features=latent_dim, out_features=latent_dim)

        # decoder
        self.decoder_l1 = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.decoder_l2 = nn.Linear(in_features=latent_dim, out_features=input_shape)
    
    def forward(self, x):
        '''
        x is the input data
        returns the reconstruction
        '''
        latent = self.run_encoder(x)
        x_hat = self.run_decoder(latent)
        return x_hat

    def run_encoder(self, x):
        output = F.relu(self.encoder_l1(x)) # relu adds non linearity
        latent = F.relu(self.encoder_l2(output)) # results in the latent vector
        return latent

    def run_decoder(self, latent):
        output = F.relu(self.decoder_l1(latent))
        x_hat = F.relu(self.decoder_l2(output))
        return x_hat

