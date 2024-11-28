import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.video_dataset.video_dataset import SequentialVideoDataset
from datasets.video_dataset.video_augmentations import VideoAugmentation

from utils.rnn_utils.evaluation_metrics import *
from utils.rnn_utils.visualization import *
from utils import arg_parser

from models.architectures.rnn_architectures.attention_lstm import AttentionLSTM
from models.architectures.rnn_architectures.residual_lstm import ResidualLSTM
from models.architectures.rnn_architectures.gru_model import GRUModel

## import tqdm
from tqdm import tqdm


def test(args, test_loader, model, criterion):
    pass 




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

    test_loader = DataLoader(test_subset, batch_size=args.bs, shuffle=True)

    INPUT_SIZE = 440*440*3 # WILL BE CHANGED LATER WITH RESPECT TO THE CNN MODELS OUTPUT SIZE

    model = AttentionLSTM(input_size=INPUT_SIZE, hidden_size=128, num_classes=len(classes), num_layers=1)
    model.load_state_dict(torch.load(args.model_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    test(args, test_loader, model, criterion)
    

