import argparse

def run_model():
    """
    Argument parser for a complete model runnning configuration.

    :return: Parsed arguments, including device, experiment ID, learning rate, batch size and number of epochs.
    """

    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', type=str, default='resnet50')

    # Device configuration (e.g., 'cuda:0' or 'cpu')
    parser.add_argument('--device', type=str, default='cuda:0')

    # Experiment ID for saving checkpoints and results
    parser.add_argument('--exp_id', type=str, default='exp_0')

    # Learning rate for the optimizer
    parser.add_argument('--lr', type=float, default=1e-3)

    # Batch size
    parser.add_argument('--bs', type=int, default=32)

    # Number of epochs
    parser.add_argument('--epoch', type=int, default=20)

    # Random state
    parser.add_argument('--random_state', type=int, default=42)

    # Transformations
    parser.add_argument('--transformations', type=str, nargs='+', default=['random_brightness', 'random_horizontal_flip', 'random_rotation'])

    args = parser.parse_args()
    return args