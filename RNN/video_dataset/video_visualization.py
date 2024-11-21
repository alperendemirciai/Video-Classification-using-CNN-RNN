import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def show_frames(sequence, normalize=True):
    """
    Displays each frame in a sequence one by one.

    Args:
        sequence (torch.Tensor): Tensor of shape (sequence_length, C, H, W).
        normalize (bool): Whether to denormalize frames (assumes normalization in [-1, 1]).
    """
    # Denormalize if needed
    if normalize:
        sequence = (sequence + 1) / 2  # Convert [-1, 1] to [0, 1]

    # Ensure the tensor is in the correct range [0, 1]
    sequence = torch.clamp(sequence, 0, 1)

    # Convert to numpy array with shape (sequence_length, H, W, C)
    sequence_np = sequence.permute(0, 2, 3, 1).cpu().numpy()  # (seq_len, H, W, C)

    for i, frame in enumerate(sequence_np):
        plt.imshow(frame)
        plt.axis("off")
        plt.title(f"Frame {i+1}")
        plt.show()
        input("Press Enter to continue to the next frame...")  # Wait for user input

def save_as_gif(frames, save_path, normalize=True, duration=200):
    """
    Save a sequence of frames as a GIF.

    Args:
        frames (torch.Tensor or np.ndarray): Shape (seq_length, H, W, C).
        save_path (str): Path to save the GIF.
        normalize (bool): Whether to normalize pixel values from 0-1 to 0-255.
        duration (int): Duration between frames in milliseconds.
    """
    # Convert frames to PIL Images
    images = []
    for frame in frames:
        if normalize:
            frame = (frame * 255).astype(np.uint8)
        images.append(Image.fromarray(frame))
    
    # Save as a GIF
    images[0].save(save_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    print(f"GIF saved at {save_path}")

