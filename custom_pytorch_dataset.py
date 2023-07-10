from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image


class Images(Dataset):

    def __init__(self):
        super().__init__()
        #Load in the dataset and assign to a dataframe
        self.df = pd.read_csv('training_data.csv')
        #Add transforms to the data which will allow it to be used in future models.
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

    def __getitem__(self, index):
            #assigning labels as features for the final dataset which will be used to train the model
            labels = self.df.loc[index,'labels']
            image_id = self.df.loc[index,'image_id']
            emotions = self.df.loc[index,'emotions']
            file_path = f'/Users/rishanrahman/Desktop/facial_emotional_recognition/Images/Training/Training/{emotions}'

            with Image.open(f'{file_path}/{image_id}') as img:
                 img.load()
                 features = self.transform(img)
                 return features,labels
                 
    def __len__(self):    
        return len(self.df)