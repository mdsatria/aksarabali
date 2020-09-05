import pandas as pd 
import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class AksaraBali(Dataset):
    def __init__(self, csv_file, root_dir, height=224, width=224, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.height = height
        self.width = width
       
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
        image = Image.open(img_path).resize((self.height, self.width))
        y_label = torch.tensor(self.annotations.iloc[index,2], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return(image, y_label)
    
def check_accuracy(loader, model, acc_type):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'{acc_type} got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channel = 3 
num_classes = 133
learning_rate = 0.001
batch_size = 64
num_epoch = 1

# tranform
my_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5005, 0.4670, 0.4174), std=(0.0813, 0.0534, 0.0932))])

train_set = AksaraBali(csv_file="../data/Train Augmix/train_label.csv",
                       root_dir="../data/Train Augmix/",
                       transform=my_transform)

test_set = AksaraBali(csv_file="../data/Test/test_label.csv",
                       root_dir="../data/Test/",
                       transform=my_transform)

#train_set = torch.utils.data(dataset)
train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_set,
                          shuffle=False,
                          batch_size=batch_size)
#test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

model = torchvision.models.resnet50(pretrained=True)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    losses = []
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # get data to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()
    
    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')
    check_accuracy(train_loader, model, "Train accuracy : ")
    check_accuracy(test_loader, model, "Test accuracy : ")
   
#torch.save(model, "model1810.pt")