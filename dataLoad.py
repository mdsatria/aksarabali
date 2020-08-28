import os
import pandas as pd
import torch
from sklearn import preprocessing

from PIL import Image
from torch.utils.data import Dataset

class AksaraBali(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform
        labels = self.annotations[1]
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(labels.astype(str))
        self.label = (targets)
       
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
        image = Image.open(img_path).resize((224, 224))
        y_label = torch.tensor(self.label[index], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return(image, y_label)