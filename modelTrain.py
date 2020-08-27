import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataLoad import AksaraBali

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channel = 3 
num_classes = 132
learning_rate = 0.001
batch_size = 16
num_epoch = 10

dataset = AksaraBali(csv_file="df_full.csv", root_dir="full", transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [20000, 6600])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

model = torchvision.models.resnet50(pretrained=False)
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