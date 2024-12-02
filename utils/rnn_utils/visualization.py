import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

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

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
    
    if not os.path.exists(plot_dir):
        os.makedirs(os.path.dirname(plot_dir), exist_ok=True)

    plt.savefig(os.path.join(plot_dir, f'{metric}_plot.png'))
    plt.close()


def plot_confusion_matrix(cm, classes, save_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    Plots the confusion matrix.
    
    Args:
    - cm (numpy.ndarray): Confusion matrix.
    - classes (list): List of class names.
    - save_path (str): The path where the plot will be saved.
    - normalize (bool): If True, normalize the confusion matrix.
    - title (str): The title of the plot.
    - cmap (matplotlib.colors.Colormap): The color map.
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if os.path.exists(save_path):
        os.remove(save_path)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    plt.close()
