import os
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

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

def score_function(engine):
    val_loss = engine.state.metrics["loss"]
    return -val_loss


# Set Training Parameters    
in_channel = 3 
num_classes = 133
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# tranform
my_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5005, 0.4670, 0.4174), 
                                   std=(0.0813, 0.0534, 0.0932))])

train_set = AksaraBali(csv_file="../data/Train Augmix/train_label.csv",
                       root_dir="train",
                       transform=my_transform)

test_set = AksaraBali(csv_file="../data/Test/test_label.csv",
                       root_dir="test",
                       transform=my_transform)

#train_set = torch.utils.data(dataset)
train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_set,
                          shuffle=False,
                          batch_size=batch_size)
#test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Inisialisasi objek GPU
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pakai model RESNET untuk transfer learning
model = torchvision.models.resnet50(pretrained=True)
model.to(gpu)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

trainer = create_supervised_trainer(model, optimizer, criterion, device=gpu)
metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}
train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=gpu)
val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=gpu)
training_history = {"accuracy":[], "loss":[]}
validation_history = {"accuracy":[], "loss":[]}
last_epoch = []

# RunningAverage metrics
RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

# EarlyStopping Callbacks
handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, handler)

# Buat Custom Function
# Custom function dibuat untuk menghubungkan dengan dua event yaitu, event saat training dan event saat evaluation.
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    accuracy = metrics["accuracy"] * 100
    loss = metrics["loss"]
    last_epoch.append(0)
    training_history["accuracy"].append(accuracy)
    training_history["loss"].append(loss)
    print("Training Results - Epochs: {} Avg. Accuracy: {:.2f} Avg. Loss: {:.2f}"
        .format(trainer.state.epoch, accuracy, loss))
  
def log_validation_results(trainer):
    val_evaluator.run(test_loader)
    metrics = val_evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    loss = metrics['loss']
    validation_history['accuracy'].append(accuracy)
    validation_history['loss'].append(loss)
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(trainer.state.epoch, accuracy, loss))
  
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

# Model Checkpoint
checkpointer = ModelCheckpoint("../Saved Models", "aksarabali_resnet", n_saved=5, 
                               create_dir=True, save_as_state_dict=True, 
                               require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {"resnet": model})

# Jalankan proses training dan evaluasi
trainer.run(train_loader, max_epochs=num_epochs)


# Plot Accuracy
plt.plot(training_history["accuracy"], label="Training Accuracy")
plt.plot(validation_history["accuracy"], label="Validation Accuracy")
plt.xlabel("Num. of Epochs")
plt.ylabel("Accuracy")
plt.legend(frameon=False)
plt.show()

# Plot Loss
plt.plot(training_history["loss"], label="Training Accuracy")
plt.plot(validation_history["loss"], label="Validation Accuracy")
plt.xlabel("Num. of Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.show()