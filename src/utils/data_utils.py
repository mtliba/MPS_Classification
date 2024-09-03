# src/grad_cam.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize the Grad-CAM object.

        Args:
            model (torch.nn.Module): The pretrained model.
            target_layer (torch.nn.Module): The target layer to visualize.
        """
        self.model = model
        self.target_layer = target_layer
        self.model.eval()

        # Hooks for gradients and activations
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Register hooks to capture gradients and activations.
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def compute_cam(self, input_tensor, class_idx=None):
        """
        Compute the Grad-CAM heatmap.

        Args:
            input_tensor (torch.Tensor): The input image tensor.
            class_idx (int, optional): The class index for which Grad-CAM is computed.

        Returns:
            np.ndarray: The computed Grad-CAM heatmap.
        """
        # Forward pass
        output = self.model(input_tensor)

        # If class index is not specified, use the predicted class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze().cpu().numpy()

        # Normalize the heatmap
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize between 0 and 1
        return cam

    def overlay_cam(self, input_image, cam):
        """
        Overlay the Grad-CAM heatmap on the input image.

        Args:
            input_image (PIL.Image or np.ndarray): The input image.
            cam (np.ndarray): The computed Grad-CAM heatmap.

        Returns:
            np.ndarray: The input image with the Grad-CAM heatmap overlay.
        """
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        overlay_image = heatmap + np.float32(input_image) / 255
        overlay_image = np.uint8(255 * overlay_image / np.max(overlay_image))

        return overlay_image
