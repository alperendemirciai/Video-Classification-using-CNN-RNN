import cv2
import numpy as np
import random
from skimage.util import random_noise

class VideoAugmentation:
    def __init__(self, random_state=None):
        if random_state is None:
            random_state = 42
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
            
    def random_rotation(self, frames, angle_range=(-30, 30)):
        angle = random.uniform(*angle_range)
        rotated_frames = []
        for frame in frames:
            (h, w) = frame.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_frame = cv2.warpAffine(frame, rotation_matrix, (w, h))
            rotated_frames.append(rotated_frame)
        return np.array(rotated_frames)

    def random_horizontal_flip(self, frames):
        flip = random.choice([True, False])
        if flip:
            flipped_frames = [cv2.flip(frame, 1) for frame in frames]
        else:
            flipped_frames = frames
        return np.array(flipped_frames)

    def random_vertical_flip(self, frames):
        flip = random.choice([True, False])
        if flip:
            flipped_frames = [cv2.flip(frame, 0) for frame in frames]
        else:
            flipped_frames = frames
        return np.array(flipped_frames)

    def random_crop(self, frames, crop_size=(100, 100)):
        crop_h, crop_w = crop_size
        cropped_frames = []
        for frame in frames:
            h, w, _ = frame.shape
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            cropped_frame = frame[top:top + crop_h, left:left + crop_w]
            cropped_frames.append(cropped_frame)
        return np.array(cropped_frames)

    def random_scale(self, frames, scale_range=(0.5, 2.0)):
        scale = random.uniform(*scale_range)
        scaled_frames = []
        for frame in frames:
            height, width = frame.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)
            scaled_frame = cv2.resize(frame, (new_width, new_height))
            scaled_frames.append(scaled_frame)
        return np.array(scaled_frames)

    def random_color_jitter(self, frames, brightness=0.2, contrast=0.2, saturation=0.2):
        jittered_frames = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hsv[..., 0] = hsv[..., 0] * (1 + saturation * random.uniform(-1, 1))
            hsv[..., 1] = hsv[..., 1] * (1 + contrast * random.uniform(-1, 1))
            hsv[..., 2] = hsv[..., 2] * (1 + brightness * random.uniform(-1, 1))
            jittered_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            jittered_frames.append(jittered_frame)
        return np.array(jittered_frames)

    def random_gaussian_noise(self, frames, var_range=(0.01, 0.05)):
        noise_var = random.uniform(*var_range)
        noisy_frames = []
        for frame in frames:
            noisy_frame = random_noise(frame, mode='gaussian', var=noise_var)
            noisy_frame = (255 * noisy_frame).astype(np.uint8)
            noisy_frames.append(noisy_frame)
        return np.array(noisy_frames)

    def random_brightness(self, frames, factor_range=(0.5, 1.5)):
        factor = random.uniform(*factor_range)
        brightened_frames = []
        for frame in frames:
            brightened_frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
            brightened_frames.append(brightened_frame)
        return np.array(brightened_frames)

    def random_contrast(self, frames, factor_range=(0.5, 1.5)):
        factor = random.uniform(*factor_range)
        contrasted_frames = []
        for frame in frames:
            mean = np.mean(frame)
            contrasted_frame = np.clip((frame - mean) * factor + mean, 0, 255).astype(np.uint8)
            contrasted_frames.append(contrasted_frame)
        return np.array(contrasted_frames)

    def random_gamma_correction(self, frames, gamma_range=(0.5, 2.0)):
        gamma = random.uniform(*gamma_range)
        gamma_corrected_frames = []
        for frame in frames:
            frame = np.float32(frame) / 255.0  # Convert to float in [0, 1]
            frame = np.clip(np.power(frame, gamma), 0, 1)  # Apply gamma correction
            frame = np.uint8(frame * 255)  # Convert back to [0, 255]
            gamma_corrected_frames.append(frame)
        return np.array(gamma_corrected_frames)
