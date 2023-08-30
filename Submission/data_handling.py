from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms as tvtransforms
import torch

'''
From this dataset, I'd like to store the images that can be accessed with their corresponding indices.
I'd also like to access images based on their labels (i.e. only images of Barack Obama)
'''
class PublicFigureDataset(Dataset):
    def __init__(self, annotations_file, img_directory, name="", transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_directory
        self.transform = transform
        self.target_transform = target_transform
        self.name = name

    def __len__(self):
        if(len(self.name) != 0):
            self.img_labels = self.img_labels.loc[self.img_labels['Name'] == self.name]
        return len(self.img_labels)

    def __getitem__(self, index):
        if(len(self.name) != 0):
            self.img_labels = self.img_labels.loc[self.img_labels['Name'] == self.name]
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 1]) # Tags images with their corresponding filenames
        image = read_image(img_path)#.type(torch.FloatTensor)
        #image = image.type(torch.FloatTensor)
        label = self.img_labels.iloc[index, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def unpeel(datadir):
    '''
    Moves all files in a directory's directories into the original directory.
    '''
    for root, dirs, files in os.walk(datadir):
        if(root == datadir): # Skips the first directory, which holds no images
            continue
        for i in range(len(files)):
            os.rename(root + '/' + files[i], datadir + '/' + files[i])

def createCSV(datadir):
    annotations = []
    for root, dirs, files in os.walk(datadir):
        if(root == datadir): # Skips the first directory
            continue
        for i in range(len(files)):
            label = os.path.basename(os.path.normpath(root))
            filename = files[i]
            annotations.append({"Name": label, "imgName": filename})

    df = pd.DataFrame(annotations)
    df.to_csv("./annotations.csv", index=False)

    return

def uniqueRename(datadir):
    '''
    Files in the dataset have reoccurring names and this method renames all of the files.
    '''
    count = 0
    for root, dirs, files in os.walk(datadir):
        if(root == datadir): # Skips the first directory, which holds no images
            continue
        for i in range(len(files)): # For each file in the directory, rename the file
            newfile = '{}x.jpg'.format(count)
            os.rename(root + '/' + files[i] ,root + '/' + newfile)
            count += 1

if __name__ == "__main__":
    DATASET_DIRECTORY = './CelebDataProcessed'
    ANNOTATIONS_DIRECTORY = "./annotations.csv"
    df = pd.read_csv(ANNOTATIONS_DIRECTORY)

    df = df.loc[df['Name'] == "William Macy"]
    print(df.iloc[0, 1])

    pubfig = PublicFigureDataset(ANNOTATIONS_DIRECTORY, DATASET_DIRECTORY)
    print(len(pubfig))


    pubfig = PublicFigureDataset(ANNOTATIONS_DIRECTORY, DATASET_DIRECTORY, name="Donald Trump")
    print(len(pubfig))