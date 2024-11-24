import os

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.transforms import Normalize

from .image_augmentation import ImageAugmentation

class ImageDataset(Dataset):
    """PyTorch dataset for image data."""

    def __init__(self, path, classes, transformations: list | None, target_size=(220, 220), mode='train', random_state=42):
        """
        :param path: Path to the dataset directory.
        :param classes: List of class names.
        :param transformations: List of image transformations to apply.
        :param target_size: Target size for each image (H, W).
        :param mode: 'train' mode for applying data augmentations. Possible values: 'train', 'valid', 'test'.
        :param random_state: Random seed for reproducibility.
        """
        self.path = path
        self.classes = classes
        self.transformations = transformations if transformations is not None else []
        self.target_size = target_size
        self.mode = mode
        self.random_state = random_state
        self.samples = self.load_samples_from_mode_()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_samples_from_mode_(self):
        samples = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.path, self.mode, class_name)
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                samples.append((image_path, label))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __num_classes__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.target_size, Image.LANCZOS)

        if self.mode == 'train':
            for transform in self.transformations:
                image = transform(image)

        image = ToTensor()(image)
        image = self.normalize(image)
        
        return image, label

# Example usage:
if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'data')
    classes = [
        'archery', 
        'baseball', 
        'basketball', 
        'bmx', 
        'bowling', 
        'boxing', 
        'cheerleading', 
        'football', 
        'golf', 
        'hammer throw', 
        'hockey', 
        'javelin',
        'olympic wrestling',
        'pole vault', 
        'rowing', 
        'skating', 
        'ski jumping', 
        'swimming', 
        'tennis', 
        'volleyball', 
        'weightlifting'
    ]
    
    ig = ImageAugmentation()
    transformations = [ig.random_gamma_correction]
    train = ImageDataset(
        path, 
        classes, 
        transformations, 
        mode='train'
    )
    test = ImageDataset(
        path, 
        classes, 
        transformations, 
        mode='test'
    )
    valid = ImageDataset(
        path, 
        classes, 
        transformations, 
        mode='valid'
    )

    print(train.__len__())
    print(test.__len__())
    print(valid.__len__())

    print(train.__getitem__(0))
    print(test.__getitem__(0))
    print(valid.__getitem__(0))

    print(train.__num_classes__())
    print(test.__num_classes__())
    print(valid.__num_classes__())