'''This module takes provides us with the sub folders containing our images. To make it compatible
with PyTorch, this module creates a dataframe containing the image id of each image as well as encoding the unique categoires'''

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Stating the route directory which contains images
folder_dir = '/Users/rishanrahman/Desktop/facial_emotional_recognition/Images/Training/Training'

#Initialising empty lists which will have image id's and emotions appended to them
image_ids = []
emotions = []

#Looping through folders in folder_dir and appending image_id's/emotions to the empty lists above
for i in os.listdir(folder_dir):
    if i != '.DS_Store':
        emotion_folder_path = f'{folder_dir}/{i}'
        for j in os.listdir(emotion_folder_path):
            emotion = os.path.basename(emotion_folder_path)
            image_ids.append(j)
            emotions.append(emotion)

d = {'image_id':image_ids,'emotions':emotions}
df = pd.DataFrame(d)

encoder = LabelEncoder()

df['labels'] =encoder.fit_transform(df['emotions']) 
