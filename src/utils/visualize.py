# src/visualize_grad_cam.py
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models import ClassificationModel
from grad_cam import GradCAM

# Load the pretrained model
def load_model(model_name, num_classes, use_sigmoid):
    model = ClassificationModel(model_name, num_classes, use_sigmoid)
    model_path = f'results/{model_name}_best.pth'  # Update this to match your saved model paths
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, input_tensor

# Visualize Grad-CAM
def visualize_grad_cam(model, image, input_tensor, target_layer):
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.compute_cam(input_tensor)
    overlay_image = grad_cam.overlay_cam(image, cam)

    # Display the original image and Grad-CAM overlay
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_image)
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == '__main__':
    model_name = 'vit_small_softmax'  # Choose your model: 'vit_small', 'resnet_small', 'efficientnet_small'
    num_classes = 2  # Adjust based on your dataset
    use_sigmoid = False  # Set to True if using independent sigmoid classifiers

    # Load the pretrained model
    model = load_model(model_name, num_classes, use_sigmoid)

    # Specify the target layer for Grad-CAM
    if model_name.startswith('vit'):
        target_layer = model.base_model.blocks[-1].norm1  # ViT last block normalization layer
    elif model_name.startswith('resnet'):
        target_layer = model.base_model.layer4[1].conv2  # ResNet last convolutional layer
    elif model_name.startswith('efficientnet'):
        target_layer = model.base_model.conv_head  # EfficientNet head convolutional layer

    # Load the image
    image_path = 'data/sample_image.jpg'  # Replace with your image path
    image, input_tensor = load_image(image_path)

    # Visualize Grad-CAM
    visualize_grad_cam(model, image, input_tensor, target_layer)
