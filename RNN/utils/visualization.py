import matplotlib.pyplot as plt
import numpy as np
import os

def plot_train_val_history(train_loss_history, val_loss_history, save_path):
    '''
    Plots the training and validation loss history over epochs.
    
    Args:
    - train_loss_history (list): List of training loss values over epochs.
    - val_loss_history (list): List of validation loss values over epochs.
    - save_path (str): The path where the plot will be saved.
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss', color='blue')
    plt.plot(val_loss_history, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_metric(metric_history, label, plot_dir, metric):
    '''
    Plots the metric history over epochs.
    
    Args:
    - metric_history (list): List of metric values over epochs.
    - label (str): The label for the metric.
    - plot_dir (str): The directory path where the plot will be saved.
    - args (argparse.Namespace): The command-line arguments.
    - metric (str): The metric name.
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(metric_history, label=label, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{label} Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'{metric}_plot.png'))
    plt.close()
