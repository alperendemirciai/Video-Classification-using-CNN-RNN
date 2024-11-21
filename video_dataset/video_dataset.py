import os
from typing import List
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomRotation
from PIL import Image
import cv2

from video_augmentations import VideoAugmentation
import video_visualization as vv


class SequentialVideoDataset(Dataset):
    def __init__(self, data_dir, classes, sequence_length=16, target_size=(220, 220), transformations:List=None, mode='train'):
        """
        Custom PyTorch dataset for loading sequential video frames.

        Args:
        - data_dir (str): Directory containing class folders, each containing video files.
        - classes (list): List of class names (folder names in data_dir).
        - sequence_length (int): Number of frames to sample from each video.
        - target_size (tuple): Target size for each frame (H, W).
        - transformations (list): List of video transformations to apply. Exp: transformations=[DA.random_horizontal_flip, DA.random_rotation]
        - mode (str): 'train' mode for applying data augmentations.
        
        """
        self.data_dir = data_dir
        self.classes = classes
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.samples = self._load_samples()
        self.mode = mode

        if transformations is None or len(transformations)==0:
            self.transformations = list()
        else:
            self.transformations = transformations

    def _load_samples(self):
        """Loads video paths and their corresponding labels."""
        samples = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            for video_file in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_file)
                samples.append((video_path, label))
        return samples

    def _read_video_frames(self, video_path):
        """Reads frames from a video file using OpenCV, sequentially."""
        frames = []
        cap = cv2.VideoCapture(video_path)  # Open the video file
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
        
        while True:
            ret, frame = cap.read()  # Read a frame
            if not ret:
                break  # Break if no more frames

            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()  # Release the video capture object
        return frames

    def _pad_to_square(self, pil_image):
        width, height = pil_image.size
        if width == height:
            return pil_image  # No padding needed if already square
        
        # Find the size of the square (the larger of the width or height)
        size = max(width, height)
        
        # Calculate the padding for each side to make the image square
        padding_left = (size - width) // 2
        padding_right = size - width - padding_left
        padding_top = (size - height) // 2
        padding_bottom = size - height - padding_top
        
        # Add padding to the image (with black padding)
        padded_image = Image.new("RGB", (size, size), (0, 0, 0))
        padded_image.paste(pil_image, (padding_left, padding_top))
        
        return padded_image


    def _sample_frames(self, frames):
        """Samples frames sequentially at regular intervals."""
        frame_count = len(frames)
        step = frame_count // self.sequence_length
        
        sampled_frames = []
        for i in range(self.sequence_length):
            index = i * step
            if index < frame_count:
                sampled_frames.append(frames[index])
            else:
                # If not enough frames, pad by repeating the last frame
                sampled_frames.append(frames[-1])
        return sampled_frames

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._read_video_frames(video_path)

        # Sample frames sequentially (at regular intervals)
        frames = self._sample_frames(frames)

        # Pad each frame to square and resize
        frames_resized = []
        for frame in frames:
            pil_image = Image.fromarray(frame)
            pil_image = self._pad_to_square(pil_image)  # Pad to square
            pil_image = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)  # Resize to target size
            frames_resized.append(pil_image)

        if self.mode == 'train':
            # Applying the transformations to the frames sequentially
            frames_array = [np.array(frame) for frame in frames_resized]
            for transform in self.transformations:
                frames_array = transform(frames_array)
            frames_resized = [Image.fromarray(frame) for frame in frames_array]
            

        # Convert frames to tensors
        frames_tensor = torch.stack([ToTensor()(frame) for frame in frames_resized])  # Shape: (sequence_length, C, H, W)
        return frames_tensor, label

    def __len__(self):
        return len(self.samples)
    
    def __num_classes__(self):
        return len(self.classes)



# Example usage
if __name__ == "__main__":
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
    
    sequence_length = 16
    target_size = (220, 220)  # Resize frames to 128x128 after padding

    va = VideoAugmentation(random_state=42)

    transformations = [va.random_brightness, va.random_color_jitter, va.random_horizontal_flip, va.random_rotation]

    dataset = SequentialVideoDataset(
        data_dir=data_dir, 
        classes=classes, 
        sequence_length=sequence_length, 
        target_size=target_size, mode='train', transformations=transformations
    )

    # Get a sample sequence
    frames, label = dataset[601]  # Get the first video and its label
    print(f"Video label: {label}, Frames shape: {frames.shape}")
    
    """
    ## Code for saving the output as a GIF

    frames_array = frames.permute(0, 2, 3, 1).numpy()  # Convert tensor to NumPy array
    # Visualize the frames
    vv.save_as_gif(frames_array, "output/sample_video6_2.gif")
    """
