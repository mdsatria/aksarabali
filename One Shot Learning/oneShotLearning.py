import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

'''
Dataset Loader
'''
class AksaraBali(Dataset):
    def __init__(self, csv_file, root_dir, setSize, transform=None, height=105, width=105):        
        csvFile = pd.read_csv(csv_file, header=None)
        self.csv_file = csvFile
        self.root_dir = root_dir
        self.setSize = setSize
        self.transform = transform
        self.categories = csvFile[2].unique()
        self.height = height
        self.width = width
    
    def __len__(self):
        return self.setSize

    def __getitem__(self, index):
        img1 = None
        img2 = None
        label = None

        if index % 2 == 0:
            category = random.choice(self.categories)
            img1_path = os.path.join(self.root_dir, random.choice(self.csv_file[self.csv_file[2]==category][0].values))
            img2_path = os.path.join(self.root_dir, random.choice(self.csv_file[self.csv_file[2]==category][0].values))
            img1 = Image.open(img1_path).resize((self.height, self.width))
            img2 = Image.open(img2_path).resize((self.height, self.width))
            label = 1.0

        else:
            category1, category2 = random.choice(self.categories), random.choice(self.categories)
            img1_path = os.path.join(self.root_dir, random.choice(self.csv_file[self.csv_file[2]==category1][0].values))
            img2_path = os.path.join(self.root_dir, random.choice(self.csv_file[self.csv_file[2]==category2][0].values))
            while img1_path == img2_path:
                img2_path = os.path.join(self.root_dir, random.choice(self.csv_file[self.csv_file[2]==category2][0].values))
            img1 = Image.open(img1_path).resize((self.height, self.width))
            img2 = Image.open(img2_path).resize((self.height, self.width))
            label = 0.0
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))

class NWayOneShotEvalSet(Dataset):
    def __init__(self, csv_file, root_dir, setSize, numWay, transform=None,height=105, width=105):
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.setSize = setSize
        self.transform = transform
        self.categories = self.csv_file[2].unique()
        self.height = height
        self.width = width
        self.numWay = numWay

    def __len__(self):
        return self.setSize

    def __getitem__(self, idx):
        # find one main image
        category =  random.choice(self.categories) # find main class label
        imgDir = os.path.join(self.root_dir, random.choice(self.csv_file[self.csv_file[2]==category][0].values)) # find random img in main class label
        mainImg = Image.open(imgDir).resize((self.height, self.width)) # open img
        if self.transform:
            mainImg = self.transform(mainImg)
        
        # find n numbers of distinct images, 1 in the same set as the main
        testSet = []
        label = np.random.randint(self.numWay)
        for i in range(self.numWay):
            if i == label:
                imgDirTest = os.path.join(self.root_dir, random.choice(self.csv_file[self.csv_file[2]==category][0].values))
                while imgDir == imgDirTest:
                    imgDirTest = os.path.join(self.root_dir, random.choice(self.csv_file[self.csv_file[2]==category][0].values))
            else:
                categoryTest = random.choice(self.categories)
                while category == categoryTest:
                    categoryTest = random.choice(self.categories)
                imgDirTest = os.path.join(self.root_dir, random.choice(self.csv_file[self.csv_file[2]==categoryTest][0].values))
            testImg = Image.open(imgDirTest).resize((self.height, self.width))
            if self.transform:
                testImg = self.transform(testImg)
            testSet.append(testImg)
        
        return mainImg, testSet, torch.from_numpy(np.array([label], dtype= int))        
    
'''
Model
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # conv2d(input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv2d(3, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fcOut = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()
        
        
    def convs(self, x):
        # out_dim = in_dim - kernel_size + 1
        # 1, 105, 105
        x = F.relu(self.bn1(self.conv1(x)))
        # 64, 96, 96
        x = F.max_pool2d(x, (2,2))
        # 64, 48, 48
        x = F.relu(self.bn2(self.conv2(x)))
        # 128, 42, 42
        x = F.max_pool2d(x, (2,2))
        # 128, 21, 21
        x = F.relu(self.bn3(self.conv3(x)))
        # 128, 18, 18
        x = F.max_pool2d(x, (2,2))
        # 128, 9, 9
        x = F.relu(self.bn4(self.conv4(x)))
        # 256, 6, 6
        return x
    
    
    def forward(self, x1, x2):
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256 * 6 * 6)
        x1 = self.sigmoid(self.fc1(x1))
        x2 = self.convs(x2)
        x2 = x2.view(-1, 256 * 6 * 6)
        x2 = self.sigmoid(self.fc1(x2))
        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x

def count_parameters(model):
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model architecture :\n\n', model)
    print(f'\nThe model has {temp:,} trainable parameters')
    
'''
Train and Eval Setting
'''
    
def train(model, train_loader, val_loader, num_epochs, criterion, save_name):
    best_val_loss = float("Inf")
    train_losses = []
    val_losses = []
    cur_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        print("Starting epoch " + str(epoch+1))
        for img1, img2, labels in train_loader:
            
            # Forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        val_running_loss = 0.0
        
        # check validation loss after every epoch
        with torch.no_grad():
            model.eval()
            for img1, img2, labels in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)
                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print('Epoch [{}/{}], Train Loss : {:.4f}, Valid Loss: {:.8f}'.format(epoch+1, num_epochs, avg_train_loss, avg_val_loss))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(save_name, model, optimizer, best_val_loss)
    print('Finished Training')
    return train_losses, val_losses

def eval(model, test_loader):
    with torch.no_grad():
        model.eval()
        correct = 0
        print('Starting Iteration')
        count = 0
        for mainImg, imgSets, label in test_loader:
            mainImg = mainImg.to(device)
            predVal = 0
            pred = -1
            for i, testImg in enumerate(imgSets):
                testImg = testImg.to(device)
                output = model(mainImg, testImg)
                if output > predVal:
                    pred = i
                    predVal = output
            label = label.to(device)
            if pred == label:
                correct += 1
            count += 1
            if count % 20 == 0:
                print(f'Current count is : {count}')
                print(f'Accuracy on n way : {correct/count}')

'''
Model Checkpoints
'''
def save_checkpoint(save_path, model, optimizer, val_loss):
    if save_path==None:
        return
    save_path = save_path
    state_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss}
    torch.save(state_dict, save_path)

    print(f'Model saved to ==> {save_path}')

def load_checkpoint(model, optimizer, save_path):
    # save_path = f'siameseNet-batchnorm50.pt'
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print(f'Model loaded from <== {save_path}')

    return val_loss

'''
Train DataLoad
'''
TrainRoot_dir = '../data/Train Original/img/'
TrainCsv_file = '../data/Train Original/trainOri_label.csv'

dataSize = 11710
TRAIN_PCT = 0.8
train_size = int(dataSize * TRAIN_PCT)
val_size = dataSize - train_size

tranformations = transforms.Compose([transforms.ToTensor()])

aksaraDataset = AksaraBali(TrainCsv_file, TrainRoot_dir, dataSize, tranformations)
train_set, val_set = random_split(aksaraDataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

print('Training data succesfully loaded')


'''
Test DataLoad
'''
TestRoot_dir = '../data/Test/img/'
TestCsv_file = '../data/Test/test_label.csv'

testSize = 7673
numWay = 132

test_set = NWayOneShotEvalSet(TestCsv_file, TestRoot_dir, testSize, numWay, tranformations)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

print('Testing data succesfully loaded')


'''
Model Compilation
'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
siameseBaseLine = Net()
siameseBaseLine = siameseBaseLine.to(device)

print(f'Model succecfully created and loaded to GPU')

'''
Model Training
'''
optimizer = optim.Adam(siameseBaseLine.parameters(), lr=0.0001)
num_epochs = 100
criterion = nn.BCEWithLogitsLoss()
save_path = 'test.pt'
train_losses, val_losses = train(siameseBaseLine, train_loader, val_loader, num_epochs, criterion, save_path)

'''
Model Validation
'''
load_model = Net().to(device)
load_optimizer = optim.Adam(load_model.parameters(), lr=0.0001)

num_epochs = 10
eval_every = 1000
total_step = len(train_loader)*num_epochs
best_val_loss = load_checkpoint(load_model, load_optimizer, save_path)

print(best_val_loss)
eval(load_model, test_loader)
# plotting of training and validation loss
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.show()