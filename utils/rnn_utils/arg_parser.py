import argparse 


def train_arg_parser():
    '''
    Argument parser for training configuration.
    
    Returns:
    - args: Parsed arguments, including device, experiment ID, learning rate, batch size, number of epochs, and mode.
    '''
    parser = argparse.ArgumentParser()
    
    # Device configuration (e.g., 'cuda:0' or 'cpu')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    # Experiment ID for saving checkpoints and results
    parser.add_argument('--exp_id', type=str, default='exp/0')
    
    # Learning rate for the optimizer
    parser.add_argument('--lr', type=float, default=3e-3)
    
    # Batch size for training
    parser.add_argument('--bs', type=int, default=10)
    
    # Number of epochs for training
    parser.add_argument('--epoch', type=int, default=20)
    
    
    args = parser.parse_args()
    return args

def test_arg_parser():
    '''
    Argument parser for testing configuration.
    
    Returns:
    - args: Parsed arguments, including device, model path, and mode.
    '''
    parser = argparse.ArgumentParser()
    
    # Device configuration (e.g., 'cuda:0' or 'cpu')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    # Path to the saved model for testing
    parser.add_argument('--model_path', type=str, default='')
    
    # Experiment ID to use latest saved model 
    parser.add_argument('--exp_id', type=str, default='exp/0')

    # Batch size for testing
    parser.add_argument('--bs', type=int, default=10)
    
    args = parser.parse_args()
    return args
