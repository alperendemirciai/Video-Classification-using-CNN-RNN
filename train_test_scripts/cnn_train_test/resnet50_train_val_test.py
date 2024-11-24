import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from datasets.image_dataset.image_dataset import ImageDataset
from datasets.image_dataset.image_augmentation import ImageAugmentation

path = os.path.join(os.getcwd(), 'data')
classes = [
    'archery', 
    'baseball', 
    'basketball', 
    'bmx', 
    'bowling', 
    'boxing', 
    'cheerleading', 
    'football', 
    'golf', 
    'hammer throw', 
    'hockey', 
    'javelin', 
    'pole vault', 
    'rowing', 
    'skating', 
    'ski jumping', 
    'swimming', 
    'tennis', 
    'volleyball', 
    'weightlifting', 
    'olympic wrestling'
]

ig = ImageAugmentation()
transformations = [ig.random_brightness, ig.random_horizontal_flip, ig.random_rotation]

train = ImageDataset(
    path, 
    classes, 
    transformations, 
    mode='train'
)
test = ImageDataset(
    path, 
    classes, 
    transformations, 
    mode='test'
)
valid = ImageDataset(
    path, 
    classes, 
    transformations, 
    mode='valid'
)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid, batch_size=32, shuffle=False)
test_loader = DataLoader(test, batch_size=32, shuffle=False)

resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.requires_grad = False

num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, len(classes))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet50.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.001)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    train_loss, train_acc = train_epoch(resnet50, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
    
    val_loss, val_acc = validate_epoch(resnet50, valid_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(resnet50.state_dict(), 'best_model.pth')
        print("Best model saved.")

resnet50.load_state_dict(torch.load('best_model.pth'))
test_acc = test_model(resnet50, test_loader, device)
print(f"Test Accuracy: {test_acc*100:.2f}%")
