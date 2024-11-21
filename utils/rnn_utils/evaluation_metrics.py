import numpy as np

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Calculate confusion matrix for multi-class classification

    Args:
    - y_true: numpy array of true labels
    - y_pred: numpy array of predicted labels
    - num_classes: number of classes

    Returns:
    - confusion_matrix: numpy array of shape (num_classes, num_classes)
    """
    confusion_matrix = np.zeros((num_classes, num_classes))
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1
    return confusion_matrix

def precision(y_true, y_pred, num_classes):
    """
    Calculate precision for multi-class classification

    Args:
    - y_true: numpy array of true labels
    - y_pred: numpy array of predicted labels
    - num_classes: number of classes
    
    Returns:
    - precision: numpy array of shape (num_classes,)
    
    """
    confusion_matrix = confusion_matrix(y_true, y_pred, num_classes)
    precision = np.zeros(num_classes)
    for i in range(num_classes):
        precision[i] = confusion_matrix[i, i] / confusion_matrix[:, i].sum()
    return precision

def recall(y_true, y_pred, num_classes):
    """
    Calculate recall for multi-class classification

    Args:
    - y_true: numpy array of true labels
    - y_pred: numpy array of predicted labels
    - num_classes: number of classes
    
    Returns:
    - recall: numpy array of shape (num_classes,)
    
    """
    confusion_matrix = confusion_matrix(y_true, y_pred, num_classes)
    recall = np.zeros(num_classes)
    for i in range(num_classes):
        recall[i] = confusion_matrix[i, i] / confusion_matrix[i, :].sum()
    return recall

def f1_score(y_true, y_pred, num_classes):
    """
    Calculate f1_score for multi-class classification

    Args:
    - y_true: numpy array of true labels
    - y_pred: numpy array of predicted labels
    - num_classes: number of classes
    
    Returns:
    - f1_score: numpy array of shape (num_classes,)
    
    """
    precision = precision(y_true, y_pred, num_classes)
    recall = recall(y_true, y_pred, num_classes)
    return 2 * (precision * recall) / (precision + recall)

def accuracy(y_true, y_pred):
    """
    Calculate accuracy for multi-class classification

    Args:
    - y_true: numpy array of true labels
    - y_pred: numpy array of predicted labels
    
    Returns:
    - accuracy: float
    
    """
    return np.mean(y_true == y_pred)


