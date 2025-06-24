import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        return heatmap

def get_last_conv_layer(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise ValueError("No Conv2d layer found in the model")

def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, np.array(image.resize((224, 224)))

def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = heatmap_colored * alpha + image
    return np.uint8(np.clip(overlay, 0, 255))

def apply_gradcam(model_name, img_path, model_loader=None):
    """
    Apply Grad-CAM and return heatmap overlay for the given image and model.
    
    Args:
        model_name (str): Name of the model (e.g., "XceptionNet", "EfficientNet", etc.)
        img_path (str): Path to the image file
        model_loader (callable): A function to load the model architecture
    
    Returns:
        np.array: Grad-CAM heatmap overlay on original image
    """
    if model_loader is None:
        raise ValueError("You must pass a model_loader function that returns an untrained model.")
    from grad_cam import GradCAM, get_last_conv_layer, preprocess_image, overlay_heatmap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model architecture
    model = model_loader(model_name)
    
    # Load trained weights
    checkpoint_path = f"checkpoints/{model_name}_best.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.eval().to(device)

    # Get last conv layer
    target_layer = get_last_conv_layer(model)
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")

    # Prepare input
    input_tensor, orig_img = preprocess_image(img_path)
    input_tensor = input_tensor.to(device)

    # Run Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate(input_tensor)
    result = overlay_heatmap(orig_img, heatmap)

    return result
