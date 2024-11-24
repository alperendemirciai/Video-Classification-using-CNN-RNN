import cv2
import numpy as np
import random

class ImageAugmentation:
    def __init__(self, random_state=42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

    def random_rotation(self, image, angle=random.uniform(-30, 30)):
        image = np.array(image)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated_image
    
    def random_horizontal_flip(self, image):
        image = np.array(image)
        return cv2.flip(image, 1) if random.choice([True, False]) else image
    
    def random_vertical_flip(self, image):
        image = np.array(image)
        return cv2.flip(image, 0) if random.choice([True, False]) else image
    
    def random_crop(self, image, crop_size=(100, 100)):
        image = np.array(image)
        crop_h, crop_w = crop_size
        h, w = image.shape[:2]
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        cropped_image = image[top:top + crop_h, left:left + crop_w]
        return cropped_image
    
    def random_scale(self, image, scale=random.uniform(0.5, 2.0)):
        image = np.array(image)
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        scaled_image = cv2.resize(image, (new_width, new_height))
        return scaled_image
    
    def random_color_jitter(self, image, brightness=0.2, contrast=0.2, saturation=0.2):
        image = np.array(image)
        brightness_factor = 1 + brightness * random.uniform(-1, 1)
        contrast_factor = 1 + contrast * random.uniform(-1, 1)
        saturation_factor = 1 + saturation * random.uniform(-1, 1)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = hsv[..., 0] * saturation_factor
        hsv[..., 1] = hsv[..., 1] * contrast_factor
        hsv[..., 2] = hsv[..., 2] * brightness_factor
        jittered_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return jittered_image
    
    def random_gaussian_noise(self, image, var=random.uniform(0.01, 0.05)):
        image = np.array(image)
        noise = np.random.normal(0, var**0.5, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def random_brightness(self, image, brightness=1 + random.uniform(-0.2, 0.2)):
        image = np.array(image)
        image = image * brightness
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def random_contrast(self, image, contrast=1 + random.uniform(-0.2, 0.2)):
        image = np.array(image)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = (image - mean) * contrast + mean
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def random_gamma_correction(self, image, gamma=random.uniform(0.5, 2.0)):
        image = np.array(image)
        table = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
# example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    ig = ImageAugmentation()
    image = mpimg.imread('data/train/cheerleading/001.jpg')
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    image = ig.random_rotation(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image = mpimg.imread('data/train/cheerleading/001.jpg')

    image = ig.random_horizontal_flip(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image = mpimg.imread('data/train/cheerleading/001.jpg')

    image = ig.random_vertical_flip(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image = mpimg.imread('data/train/cheerleading/001.jpg')

    image = ig.random_crop(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image = mpimg.imread('data/train/cheerleading/001.jpg')

    image = ig.random_scale(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image = mpimg.imread('data/train/cheerleading/001.jpg')

    image = ig.random_color_jitter(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image = mpimg.imread('data/train/cheerleading/001.jpg')

    image = ig.random_gaussian_noise(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image = mpimg.imread('data/train/cheerleading/001.jpg')

    image = ig.random_brightness(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image = mpimg.imread('data/train/cheerleading/001.jpg')

    image = ig.random_contrast(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image = mpimg.imread('data/train/cheerleading/001.jpg')

    image = ig.random_gamma_correction(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()