import torch
import os
import pathlib
import numpy as np
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import re
import pandas as pd
import string
import math
from tqdm.auto import tqdm
from collections import Counter
from torch.utils.data import DataLoader, Subset, Dataset
from torchmetrics import MeanSquaredError, F1Score
from torch import nn
plt.style.use('ggplot')

MAX_LEN = 1024
NUM_WORDS = 32000 
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
EPOCHS = 30
LR = 0.00001

# Вспомогательная функция RMSE

def RMSE(outputs, labels):
    mse = MeanSquaredError().to(device)
    return math.sqrt(mse(outputs, labels))

# Вспомогательная функция MSE

def F1(outputs, labels, num_classes):
    if num_classes > 2:
        task = 'multiclass'
        
    else:
        task = 'binary'
        

    f1 = F1Score()

# Заглушка для модели
# array = np.random.rand(2047, 51)
# np.savetxt('заглушка.csv', array, delimiter = ',')

class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx, :].values.astype(np.float32)
        return torch.tensor(x, dtype = torch.float32)

# df = pd.read_excel('PQv2_for_pandas.xlsx')    
# df.drop(['Data'], axis = 1, inplace = True)

X = pd.read_csv('X_features.csv', header = None)
y = pd.read_excel('PQv2_for_pandas.xlsx').loc[:, 'Total_%_water']

df = pd.concat([X, y], axis = 1)

dataset = DataFrameDataset(df)

train_loader = DataLoader(dataset[:round(len(dataset) * TRAIN_SPLIT)], batch_size = BATCH_SIZE, shuffle = True)
valid_loader = DataLoader(dataset[round(len(dataset) * TRAIN_SPLIT):], batch_size = BATCH_SIZE, shuffle = False)


# Model parameters.
EMBED_DIM = 50
NUM_ENCODER_LAYERS = 3
NUM_HEADS = 5


class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super(EncoderClassifier, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, 
            nhead=num_heads, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        out = self.linear(x)
        return out   

###----------------------------------------------ТРЕНИРОВКА----------------------------------------------###

def train(model, trainloader, optimizer, criterion, device, metrics):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_metric = 0.0
   
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, labels = data[:, :-1], data[:, -1]
        inputs = inputs.to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        
        # Forward pass.
        outputs = model(inputs)
        outputs = torch.squeeze(outputs, -1)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        train_running_loss += loss.item()
        train_running_metric += metrics(outputs, labels)
        # Backpropagation.
        loss.backward()
        # Update the optimizer parameters.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / len(trainloader)
    epoch_acc = train_running_metric / len(trainloader)

    return epoch_loss, epoch_acc

###----------------------------------------------ВАЛИДАЦИЯ----------------------------------------------###

def validate(model, testloader, criterion, device, metrics):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_metric = 0.0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            
            inputs, labels = data[:, :-1], data[:, -1]
            inputs = inputs.to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            # Forward pass.
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, -1)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            #if metrics
            valid_running_metric += metrics(outputs, labels)
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / len(testloader)
    epoch_acc = valid_running_metric / len(testloader)

    return epoch_loss, epoch_acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EncoderClassifier(
    vocab_size=NUM_WORDS,
    embed_dim=EMBED_DIM,
    num_layers=NUM_ENCODER_LAYERS,
    num_heads=NUM_HEADS
).to(device)

print(model)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")



optimizer = optim.Adam(
    model.parameters(), 
    lr=LR,
)


def Pipeline(task):
    if task == 'Regression':
        metrics = RMSE
        criterion = nn.MSELoss()
    elif task == 'Classification':
        metrics = F1Score()
        
metrics = RMSE

criterion = nn.MSELoss()


###----------------------------------------------ЦИКЛ ОБУЧЕНИЯ----------------------------------------------###

# Lists to keep track of losses and accuracies.
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
least_loss = float('inf')
# Start the training.
for epoch in range(EPOCHS):
    print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                            optimizer, criterion, device, metrics)
    
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                criterion, device, metrics)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss}, training acc: {train_epoch_acc}")
    print(f"Validation loss: {valid_epoch_loss}, validation acc: {valid_epoch_acc}")
    print('-'*50)