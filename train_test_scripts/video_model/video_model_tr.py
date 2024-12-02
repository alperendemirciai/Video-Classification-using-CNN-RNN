import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datasets.video_dataset.video_dataset import SequentialVideoDataset
from datasets.video_dataset.video_augmentations import VideoAugmentation
from utils.rnn_utils.evaluation_metrics import *
from utils.rnn_utils.visualization import *
from utils.rnn_utils import arg_parser
from models.architectures.rnn_architectures.attention_lstm import AttentionLSTM
from models.architectures.rnn_architectures.residual_lstm import ResLSTM
from models.architectures.rnn_architectures.gru_model import GRUModel
from models.architectures.rnn_architectures.lstm_model import LSTMModel
from models.architectures.cnn_architectures.resnet50 import ResNet50Model

from typing import Tuple, Optional


def train(args, train_loader, val_loader, rnn, cnn, criterion, optimizer, save_path='../plots'):
    """
    Train the RNN model using extracted features from the pretrained CNN.
    """
    print(f"Training {rnn.__class__.__name__} for {args.epoch} epochs...")
    print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples.")
    print(f"Training on {args.device}...")
    ## print the total number of parameters in the model
    print(f"Total number of parameters in the model: {sum(p.numel() for p in rnn.parameters())}")
    os.makedirs(f'{save_path}/{args.exp_id}', exist_ok=True)



    ## Save the model
    os.makedirs('../checkpoints', exist_ok=True)
    os.makedirs(f'../checkpoints/{args.exp_id}', exist_ok=True)

    train_loss_history, val_loss_history, val_accuracy_history = [], [], []

    for epoch in range(args.epoch):
        train_loss, train_accuracy = train_one_epoch(args, train_loader, rnn, cnn, criterion, optimizer)
        val_loss, val_accuracy = validate_one_epoch(args, val_loader, rnn, cnn, criterion)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{args.epoch}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save intermediate checkpoints
        torch.save(rnn.state_dict(), f'../checkpoints/{args.exp_id}/epoch_{epoch + 1}.pth')

        # save the metrics to a text file
        with open(f'{save_path}/{args.exp_id}/rnn_metrics.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}/{args.epoch}:\n")
            f.write(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}\n")
            f.write(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n\n")

    # Save final model
    torch.save(rnn.state_dict(), f'../checkpoints/{args.exp_id}/final_rnn.pth')

    # Plot metrics
    plot_train_val_history(train_loss_history, val_loss_history, f'{save_path}/{args.exp_id}/rnn_train_val_loss.png')
    plot_metric(val_accuracy_history, 'Validation', f'{save_path}/{args.exp_id}', 'Accuracy')


def train_one_epoch(args, train_loader, rnn, cnn, criterion, optimizer) -> Tuple[float, float]:
    rnn.train()
    cnn.eval()  # Keep CNN frozen
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        # Only load a batch of frames at a time instead of all frames
        b, s, c, h, w = inputs.size()
        inputs = inputs.view(b * s, c, h, w)

        # Use no_grad for CNN feature extraction during training (no gradient calculation)
        with torch.no_grad():
            features = cnn.feature_extractor(inputs)
        
        features = features.view(b, s, -1)

        # Forward pass through RNN
        optimizer.zero_grad()
        outputs, _ = rnn(features)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def validate_one_epoch(args, val_loader, rnn, cnn, criterion) -> Tuple[float, float]:
    rnn.eval()
    cnn.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # Extract features with CNN (no_grad to save memory)
            b, s, c, h, w = inputs.size()
            inputs = inputs.view(b * s, c, h, w)
            features = cnn.feature_extractor(inputs)
            features = features.view(b, s, -1)

            # Forward pass through RNN
            outputs, _ = rnn(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


if __name__ == '__main__':
    args = arg_parser.train_arg_parser()
    torch.manual_seed(args.random_state)

    # Dataset and augmentation
    data_dir = "../../Videos"
    classes = [
        "archery", "baseball", "basketball", "bmx", "bowling", "boxing", "cheerleading", "golf",
        "hammerthrow", "highjump", "hockey", "hurdling", "javelin", "polevault", "rowing",
        "swimming", "tennis", "volleyball", "weight", "wrestling", "discusthrow", "skating",
        "skiing", "running", "shotput", "soccer"
    ]

    sequence_length, target_size = 16, (224, 224)

    #va = VideoAugmentation(random_state=args.random_state)
    #transformations = [va.random_brightness, va.random_horizontal_flip, va.random_rotation]

    dataset = SequentialVideoDataset(
        data_dir=data_dir, classes=classes, sequence_length=sequence_length,
        target_size=target_size, mode='train', ##transformations=transformations,
        random_state=args.random_state
    )

    train_subset, val_subset, test_subset = dataset.train_test_val_split(val_size=0.2, test_size=0.2)
    train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.bs, shuffle=False, num_workers=2)

    # Model and training setup
    INPUT_SIZE = 2048
    cnn = ResNet50Model(num_classes=len(classes), pretrained=True).to(args.device)
    ##rnn = ResLSTM(input_size=INPUT_SIZE, hidden_size=16, num_classes=len(classes), num_layers=1).to(args.device)
    ##rnn = GRUModel(input_size=INPUT_SIZE, hidden_size=16, num_classes=len(classes), num_layers=2).to(args.device)
    rnn = LSTMModel(input_size=INPUT_SIZE, hidden_size=16, num_classes=len(classes), num_layers=1, bidirectional=True).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=args.lr)

    # Start training
    train(args, train_loader, val_loader, rnn, cnn, criterion, optimizer)
