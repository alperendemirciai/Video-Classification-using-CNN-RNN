import torch
import numpy as np

def save_model(model, save_path):
    '''
    Saves the state dictionary of a PyTorch model to a specified path.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to save.
    - save_path (str): The path where the model's state dictionary will be saved.
    '''
    torch.save(model.state_dict(), save_path)

def set_seed(seed):
    '''
    Sets the random seed for reproducibility in NumPy and PyTorch.
    
    Args:
    - seed (int): The seed value for random number generators.
    
    Notes:
    - Ensures that the results are reproducible by fixing the seed for various random number generators.
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(model, load_path):
    '''
    Loads the state dictionary of a PyTorch model from a specified path.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to load the state dictionary into.
    - load_path (str): The path where the model's state dictionary is saved.
    '''
    model.load_state_dict(torch.load(load_path))

def get_device():
    '''
    Returns the device where PyTorch tensors are stored.
    
    Returns:
    - device (torch.device): The device where PyTorch tensors are stored.
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    return device
