import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.video_dataset.video_dataset import SequentialVideoDataset
from datasets.video_dataset.video_augmentations import VideoAugmentation

from utils.rnn_utils.evaluation_metrics import *
from utils.rnn_utils.visualization import *
from utils.rnn_utils import arg_parser

from models.architectures.rnn_architectures.attention_lstm import AttentionLSTM
from models.architectures.rnn_architectures.residual_lstm import ResidualLSTM
from models.architectures.rnn_architectures.gru_model import GRUModel

## import tqdm
from tqdm import tqdm


def train(args, train_loader, val_loader, model, criterion, optimizer):
    """
    Train the model for the specified number of epochs.

    Args:
    - args: Parsed arguments.
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    - model: The model to train.
    - criterion: The loss function.
    - optimizer: The optimizer.
    
    """

    print(f"Training {model.__class__.__name__} for {args.epoch} epochs...")
 
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []
    
    for epoch in range(args.epoch):
        train_loss, train_accuracy = train_one_epoch(args, train_loader, model, criterion, optimizer)
        val_loss, val_accuracy = validate_one_epoch(args, val_loader, model, criterion)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{args.epoch}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    plot_train_val_history(train_loss_history, val_loss_history, f'../plots/{args.exp_id}/rnn_train_val_loss.png')
    plot_metric(val_accuracy_history, 'Validation', f'../plots/{args.exp_id}', 'Accuracy')

def train_one_epoch(args, train_loader, model, criterion, optimizer):
    """
    Train the model for one epoch.

    Args:
    - args: Parsed arguments.
    - train_loader: DataLoader for the training set.
    - model: The model to train.
    - criterion: The loss function.
    - optimizer: The optimizer.

    Returns:
    - epoch_loss: The average loss over the epoch.
    - epoch_accuracy: The accuracy over the epoch.
    """

    train_loader = tqdm(train_loader, desc="Training")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

def validate_one_epoch(args, val_loader, model, criterion):
    """
    Validate the model for one epoch.

    Args:
    - args: Parsed arguments.
    - val_loader: DataLoader for the validation set.
    - model: The model to validate.
    - criterion: The loss function.

    Returns:
    - epoch_loss: The average loss over the epoch.
    - epoch_accuracy: The accuracy over the epoch.
    """

    val_loader = tqdm(val_loader, desc="Validating")
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            outputs = model(inputs)
            
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
    
    data_dir = "../Videos"
    classes = [
        "archery",
        "baseball",
        "basketball",
        "bmx",
        "bowling",
        "boxing",
        "cheerleading",
        "golf",
        "hammerthrow",
        "highjump",
        "hockey",
        "hurdling",
        "javelin",
        "polevault",
        "rowing",
        "swimming",
        "tennis",
        "volleyball",
        "weight",
        "wrestling",
        "discusthrow",
        "skating",
        "skiing",
        "running",
        "shotput",
        "soccer"
    ]# Class folder names
    
    sequence_length = 32
    target_size = (440, 440)  # Resize frames to 440x440 pixels

    va = VideoAugmentation(random_state=args.random_state)

    transformations = [va.random_brightness, va.random_horizontal_flip, va.random_rotation]

    dataset = SequentialVideoDataset(
        data_dir=data_dir, 
        classes=classes, 
        sequence_length=sequence_length, 
        target_size=target_size, mode='train', transformations=transformations, random_state=args.random_state
    )

    ## define the train_val split

    train_subset, val_subset, test_subset = dataset.train_test_val_split(val_size=0.2, test_size=0.15)

    train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.bs, shuffle=False)

    INPUT_SIZE = 2048 # WILL BE CHANGED LATER WITH RESPECT TO THE CNN MODELS OUTPUT SIZE

    model = AttentionLSTM(input_size=INPUT_SIZE, hidden_size=128, num_classes=len(classes), num_layers=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, train_loader, val_loader, model, criterion, optimizer)
    

