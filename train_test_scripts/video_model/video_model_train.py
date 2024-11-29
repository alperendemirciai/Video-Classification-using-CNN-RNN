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


def train(args, train_loader, val_loader, rnn, cnn, criterion, optimizer):
    """
    Train the RNN model using extracted features from the pretrained CNN.
    """
    print(f"Training {rnn.__class__.__name__} for {args.epoch} epochs...")
    print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples.")
    print(f"Training on {args.device}...")


    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []

    for epoch in range(args.epoch):
        train_loss, train_accuracy = train_one_epoch(args, train_loader, rnn, cnn, criterion, optimizer)
        #train_loss = 0.0
        #train_accuracy = 0.0
        val_loss, val_accuracy = validate_one_epoch(args, val_loader, rnn, cnn, criterion)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{args.epoch}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    
    ## Save the model
    os.makedirs('../checkpoints', exist_ok=True)
    os.makedirs(f'../checkpoints/{args.exp_id}', exist_ok=True)

    torch.save(rnn.state_dict(), f'../checkpoints/{args.exp_id}/final_rnn.pth')


    plot_train_val_history(train_loss_history, val_loss_history, f'../plots/{args.exp_id}/rnn_train_val_loss.png')
    plot_metric(val_accuracy_history, 'Validation', f'../plots/{args.exp_id}', 'Accuracy')

def train_one_epoch(args, train_loader, rnn, cnn, criterion, optimizer) -> Tuple[float, float]:
    train_loader = tqdm(train_loader, desc="Training")
    rnn.train()
    cnn.eval()  # CNN is frozen during training
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        # Process frames through CNN
        b, s, c, h, w = inputs.size()
        ##print(f"Initial inputs shape: {inputs.shape}, labels shape: {labels.shape}")  # Debug

        inputs = inputs.view(b * s, c, h, w)  # Reshape for CNN
        with torch.no_grad():
            features = cnn.feature_extractor(inputs)  # Output: (B * S, F)

        # Reshape features for RNN
        features = features.view(b, s, -1)  # Output: (B, S, F)
        ##print(f"Features shape: {features.shape}")

        optimizer.zero_grad()
        outputs, _ = rnn(features)  # Output: (B, num_classes)
        ##print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
        ##print(_.__len__(), "hidden state length")  # Debug

        loss = criterion(outputs, labels)  # Ensure shapes match
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def validate_one_epoch(args, val_loader, rnn, cnn, criterion):
    """
    Validate the RNN for one epoch using features extracted from the CNN.
    """
    val_loader = tqdm(val_loader, desc="Validating")
    rnn.eval()
    cnn.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # Process frames through CNN: (B, S, C, H, W) -> (B * S, C, H, W)
            b, s, c, h, w = inputs.size()
            inputs = inputs.view(b * s, c, h, w)
            features = cnn.feature_extractor(inputs)  # Output: (B * S, F)

            # Reshape features for RNN: (B * S, F) -> (B, S, F)
            features = features.view(b, s, -1)
            outputs,_ = rnn(features)  # Output: (B, num_classes)
            print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
            print(_.__len__(), "hidden state length")  # Debug
            print(outputs, labels)
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
    data_dir = "../../Videos"

    classes = [
        "archery", "baseball", "basketball", "bmx", "bowling", "boxing", "cheerleading", "golf",
        "hammerthrow", "highjump", "hockey", "hurdling", "javelin", "polevault", "rowing",
        "swimming", "tennis", "volleyball", "weight", "wrestling", "discusthrow", "skating",
        "skiing", "running", "shotput", "soccer"
    ]

    sequence_length = 16  # Number of frames per sequence
    target_size = (224, 224)  # Resize frames for CNN input

    va = VideoAugmentation(random_state=args.random_state)
    transformations = [va.random_brightness, va.random_horizontal_flip, va.random_rotation]

    dataset = SequentialVideoDataset(
        data_dir=data_dir,
        classes=classes,
        sequence_length=sequence_length,
        target_size=target_size,
        mode='train',
        transformations=transformations,
        random_state=args.random_state
    )

    train_subset, val_subset, test_subset = dataset.train_test_val_split(val_size=0.2, test_size=0.20)

    train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.bs, shuffle=False)

    INPUT_SIZE = 2048  # ResNet50 feature size

    cnn = ResNet50Model(num_classes=len(classes), pretrained=True).to(args.device)
    rnn = ResLSTM(input_size=INPUT_SIZE, hidden_size=16, num_classes=len(classes), num_layers=1).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=args.lr)

    train(args, train_loader, val_loader, rnn, cnn, criterion, optimizer)
