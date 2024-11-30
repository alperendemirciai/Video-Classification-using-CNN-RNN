import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from datasets.image_dataset.image_dataset import ImageDataset
from datasets.image_dataset.image_augmentation import ImageAugmentation

from utils.cnn_utils import arg_parser

from models.architectures.cnn_architectures.resnet50 import ResNet50Model
# TODO: Import other models

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

if __name__ == '__main__':
    # default run: py -m train_test_scripts.cnn_train_test.cnn_train_val_test
    args = arg_parser.run_model()

    torch.manual_seed(args.random_state)
    device = args.device
    path = os.path.join(os.getcwd(), 'data')
    model_path = os.path.join(os.getcwd(), 'models/cnn_models')

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

    ig = ImageAugmentation(random_state=args.random_state)

    transformation_map = {
        'random_rotation': ig.random_rotation,
        'random_horizontal_flip': ig.random_horizontal_flip,
        'random_vertical_flip': ig.random_vertical_flip,
        'random_crop': ig.random_crop,
        'random_scale': ig.random_scale,
        'random_color_jitter': ig.random_color_jitter,
        'random_gaussian_noise': ig.random_gaussian_noise,
        'random_brightness': ig.random_brightness,
        'random_contrast': ig.random_contrast,
        'random_gamma_correction': ig.random_gamma_correction
    }

    transformations = [transformation_map[trans] for trans in args.transformations]

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

    train_loader = DataLoader(
        train, 
        batch_size=args.bs, 
        shuffle=True,
        num_workers=4
    )
    valid_loader = DataLoader(
        valid, 
        batch_size=args.bs, 
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test, 
        batch_size=args.bs, 
        shuffle=False,
        num_workers=4
    )

    model_map = {
        'resnet50': ResNet50Model
        # TODO: Add other models
    }

    if args.model in ['resnet50']: # or other pretrained model
        model = model_map[args.model](num_classes=len(classes), pretrained=True) # TODO: Add other pretrained model
        optimizer = optim.Adam(model.resnet50.fc.parameters(), lr=args.lr)
    else:
        model = model_map[args.model](num_classes=len(classes))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    num_epochs = args.epoch
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
        
        val_loss, val_acc = validate_epoch(model, valid_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_path, f'{args.model}_{args.exp_id}.pth'))
            print("Model saved.")

    model.load_state_dict(torch.load(os.path.join(model_path, f'{args.model}_{args.exp_id}.pth')))
    test_acc = test_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
