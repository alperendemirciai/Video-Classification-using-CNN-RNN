import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomRotation
from PIL import Image
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import video_augmentations as va
import video_visualization as vv



class SequentialVideoDataset(Dataset):
    def __init__(self, data_dir, classes, sequence_length=16, target_size=(128, 128), transformations=None):
        self.data_dir = data_dir
        self.classes = classes
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.samples = self._load_samples()

        if transformations is None:
            self.transformations = Compose([
                ToTensor(),  # Convert to tensor and permute (H, W, C) -> (C, H, W)
            ])
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

            frames_resized.append(self.transformations(pil_image))
        
        # Stack frames into a single tensor
        frames_tensor = torch.stack(frames_resized)  # Shape: (sequence_length, C, H, W)
        
        return frames_tensor, label

    def __len__(self):
        return len(self.samples)
    
    def __num_classes__(self):
        return len(self.classes)



# Example usage
if __name__ == "__main__":
    data_dir = "../../Videos"
    classes = [
        "archery", "bmx", "cheerleading", "football", "hammerthrow", "hurdling",
        "polevault", "shotput", "soccer", "volleyball", "baseball", "bowling",
        "discusthrow", "golf", "highjump", "javelin", "rowing", "skating",
        "swimming", "weight", "basketball", "boxing", "diving", "gymnastics",
        "hockey", "longjump", "running", "skiing", "tennis", "wrestling"
        ] # Class folder names
    
    sequence_length = 16
    target_size = (440, 440)  # Resize frames to 128x128 after padding

    dataset = SequentialVideoDataset(
        data_dir=data_dir, 
        classes=classes, 
        sequence_length=sequence_length, 
        target_size=target_size
    )

    # Get a sample sequence
    frames, label = dataset[600]  # Get the first video and its label
    print(f"Video label: {label}, Frames shape: {frames.shape}")
    ##show_frames(frames, normalize=True)

    # Ensure frames are numpy arrays
    if isinstance(frames, torch.Tensor):
        frames = frames.permute(0, 2, 3, 1).numpy()  # Convert from (T, C, H, W) to (T, H, W, C)

    # Save as GIF
    gif_path = "./output/sample_video5.gif"
    vv.save_as_gif(frames, gif_path, normalize=True)

    frames = va.random_rotation(frames)
    frames = va.random_horizontal_flip(frames)

    vv.save_as_gif(frames, "./output/sample_video5_augmented.gif", normalize=True)


