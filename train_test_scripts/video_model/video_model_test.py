import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.video_dataset.video_dataset import SequentialVideoDataset
from models.architectures.rnn_architectures.attention_lstm import AttentionLSTM
from models.architectures.cnn_architectures.resnet50 import ResNet50Model
from utils.rnn_utils.evaluation_metrics import calculate_accuracy, confusion_matrix
from utils.rnn_utils.visualization import plot_confusion_matrix

# Parameters
DATA_DIR = "../Videos"  # Path to video dataset
CHECKPOINT_PATH = "../../checkpoints/best_model.pth"  # Path to trained model checkpoint
SEQUENCE_LENGTH = 32
TARGET_SIZE = (440, 440)
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "archery", "baseball", "basketball", "bmx", "bowling", "boxing",
    "cheerleading", "golf", "hammerthrow", "highjump", "hockey",
    "hurdling", "javelin", "polevault", "rowing", "swimming", "tennis",
    "volleyball", "weight", "wrestling", "discusthrow", "skating",
    "skiing", "running", "shotput", "soccer"
]

def test_model(test_loader, cnn, rnn, criterion):
    """
    Test the model on the test dataset.

    Args:
    - test_loader: DataLoader for the test set.
    - cnn: The pre-trained CNN model.
    - rnn: The trained RNN model.
    - criterion: The loss function.

    Returns:
    - test_loss: Average loss on the test set.
    - test_accuracy: Accuracy on the test set.
    """
    cnn.eval()
    rnn.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Pass through CNN
            batch_size, seq_len, c, h, w = inputs.shape
            inputs = inputs.view(batch_size * seq_len, c, h, w)  # Reshape for CNN
            cnn_features = cnn.feature_extractor(inputs)  # Shape: (batch_size * seq_len, feature_dim)
            cnn_features = cnn_features.view(batch_size, seq_len, -1)  # Reshape for RNN
            
            # Pass through RNN
            outputs = rnn(cnn_features)  # Shape: (batch_size, num_classes)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total
    return test_loss, test_accuracy, all_preds, all_labels

if __name__ == "__main__":
    # Load the test dataset
    dataset = SequentialVideoDataset(
        data_dir=DATA_DIR,
        classes=CLASSES,
        sequence_length=SEQUENCE_LENGTH,
        target_size=TARGET_SIZE,
        mode='test'
    )

    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load models
    INPUT_SIZE = 2048  # Adjust based on your CNN output size
    cnn = ResNet50Model(num_classes=len(CLASSES), pretrained=True)
    rnn = AttentionLSTM(input_size=INPUT_SIZE, hidden_size=128, num_classes=len(CLASSES), num_layers=1)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    rnn.load_state_dict(checkpoint['rnn_state_dict'])

    cnn.to(DEVICE)
    rnn.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Test the model
    test_loss, test_accuracy, all_preds, all_labels = test_model(test_loader, cnn, rnn, criterion)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate additional metrics
    cm = confusion_matrix(all_labels, all_preds, class_names=CLASSES)
    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix as an image
    plot_confusion_matrix(cm, CLASSES, output_path="../plots/test_confusion_matrix.png")
