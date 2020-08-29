import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataLoad import AksaraBali

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
in_channel = 3 
num_classes = 133
learning_rate = 0.001
batch_size = 64
num_epoch = 1

# tranform
my_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5005, 0.4670, 0.4174), std=(0.0813, 0.0534, 0.0932))])

train_set = AksaraBali(csv_file="label/train_label.csv",
                       root_dir="images/train",
                       transform=my_transform)

test_set = AksaraBali(csv_file="label/test_label.csv",
                       root_dir="images/test",
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