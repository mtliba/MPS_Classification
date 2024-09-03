# src/dataset.py
import os
from PIL import Image, ImageEnhance
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import random
import numpy as np
import os
import cv2

# Set the random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(42)  # Set the seed for reproducibility


class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images, organized by folders where the folder name is the label.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Load all image file paths and labels
        for label_folder in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_folder)
            if os.path.isdir(label_path):
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.png') or file_name.endswith('.jpg'):
                        self.image_files.append(os.path.join(label_path, file_name))
                        self.labels.append(int(label_folder))  # Convert the folder name to an integer label

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale (L mode)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Define a custom transformation for image sharpening
class SharpenImage:
    def __init__(self, factor=2.0):
        self.factor = factor

    def __call__(self, image):
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(self.factor)


# Define a function to get transformations with resizing, normalization, and augmentations
def get_transformations(image_size=(224, 224)):
    return {
        'Original': transforms.Compose([
            transforms.Resize(image_size),                   # Resize to the desired size
            transforms.ToTensor(),                          # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])     # Normalize to [-1, 1] range
        ]),
        'Grayscale': transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1),    # Ensure image is grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'Horizontal Flip': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'Vertical Flip': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'Rotation': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'Sharpen': transforms.Compose([
            transforms.Resize(image_size),
            SharpenImage(factor=2.0),                       # Apply image sharpening
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'Combination': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),         # Random horizontal flip
            transforms.RandomVerticalFlip(p=0.5),           # Random vertical flip
            transforms.RandomRotation(degrees=45),          # Random rotation
            SharpenImage(factor=2.0),                       # Apply image sharpening
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    }

# Custom Transformations for Medical Images
class CLAHEEqualization:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        np_image = np.array(image)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        equalized_image = clahe.apply(np_image)
        return Image.fromarray(equalized_image)


# Define a function to get transformations with resizing, normalization, and augmentations
def get_transformations_medical(image_size=(224, 224)):
    return {
        'Original': transforms.Compose([
            transforms.Resize(image_size),                   # Resize to the desired size
            transforms.ToTensor(),                          # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])     # Normalize to [-1, 1] range
        ]),
        'Histogram Equalization': transforms.Compose([
            transforms.Resize(image_size),
            CLAHEEqualization(clip_limit=2.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'Edge Detection': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.tensor(cv2.Canny(np.array(x.permute(1, 2, 0)), 100, 200)).float().unsqueeze(0)),  # Apply Canny edge detection
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'Affine Transform': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'Elastic Deformation': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    }
# Define a function to get all combined transformations
def get_combined_medical_transformations(image_size=(224, 224)):
    return transforms.Compose([
        # Resize and normalize
        transforms.Resize(image_size),
        
        # Random Horizontal and Vertical Flip
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        # Random Rotation
        transforms.RandomRotation(degrees=30),
        
        # Affine Transformations
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
        # Elastic Deformation (Perspective)
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),

        # Histogram Equalization
        CLAHEEqualization(clip_limit=2.0),

        # Edge Detection (Canny)
        transforms.Lambda(lambda x: Image.fromarray(cv2.Canny(np.array(x), 100, 200))),

        # Sharpening
        SharpenImage(factor=2.0),

        # Convert to Tensor
        transforms.ToTensor(),

        # Intensity Normalization
        transforms.Normalize(mean=[0.5], std=[0.5]),

        # Random Erasing
        transforms.RandomErasing(p=0.5),

        # Padding to maintain aspect ratio (if necessary)
        transforms.Pad(padding=10, fill=0, padding_mode='constant')
    ])


def show_and_save_images(grid, title, filename):
    plt.figure(figsize=(12, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')  # Convert CHW to HWC for plotting grayscale
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename)  # Save the plot to a file
    plt.show()

if __name__ == '__main__':
    

    # Define the directory containing your images organized by labels
    image_dir = 'data/raw/medical_images'

    # Load the dataset without transformations to get the original images
    dataset = MedicalImageDataset(root_dir=image_dir)

    # Get transformations
    transformations = get_transformations(image_size=(224, 224))


    # Get a few original images to display in the first row
    num_images = 4  # Number of images to display in the first row
    original_images, labels = zip(*[dataset[i] for i in range(num_images)])

    # Display and save the original images
    original_grid = make_grid(original_images, nrow=num_images)
    show_and_save_images(original_grid, 'Original Images', 'results/original_images.png')

    # Test the transformations and display augmented images
    for transform_name, transform in transformations.items():
        # Apply the transformation
        transformed_dataset = MedicalImageDataset(root_dir=image_dir, transform=transform)
        
        # Get transformed versions of the selected original images
        augmented_images = []
        for i in range(num_images):
            transformed_image, _ = transformed_dataset[i]
            augmented_images.append(transformed_image)

        # Create a grid with the original images in the first row and augmented images in subsequent rows
        grid_images = original_images + tuple(augmented_images)  # Combine original and augmented images
        grid = make_grid(grid_images, nrow=num_images)  # Create a grid with nrow for original images

        # Display and save the augmented images
        show_and_save_images(grid, f'Transformed: {transform_name}', f'results/{transform_name}_images.png')

