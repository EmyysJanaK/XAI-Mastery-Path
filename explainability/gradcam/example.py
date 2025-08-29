"""
Example usage of GradCAM for visualizing CNN decisions.

This script demonstrates how to use GradCAM to visualize which parts of an image
contribute most to a particular classification decision.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from gradcam import GradCAM
from gradcam_variants import GradCAMPlusPlus, SmoothGradCAM, XGradCAM
from visualization import visualize_gradcam
from layer_utils import find_target_layer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GradCAM Visualization Example")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "vgg16", "densenet121", "mobilenet_v2"],
                        help="Model architecture to use")
    parser.add_argument("--method", type=str, default="gradcam",
                        choices=["gradcam", "gradcam++", "smoothgrad", "xgradcam"],
                        help="GradCAM method to use")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class index (None for predicted class)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save visualization results")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")
    return parser.parse_args()


def load_image(image_path, size=224):
    """
    Load and preprocess an image for model input.
    
    Args:
        image_path: Path to the input image
        size: Target image size
        
    Returns:
        Tuple of (original PIL image, preprocessed tensor)
    """
    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Preprocess image
    input_tensor = preprocess(img)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    return img, input_tensor


def get_imagenet_class_name(class_idx):
    """
    Get ImageNet class name from class index.
    
    Args:
        class_idx: ImageNet class index
        
    Returns:
        Class name string
    """
    # This is a minimal mapping of some common ImageNet classes
    # In a real application, you would load the full mapping from a file
    imagenet_classes = {
        0: "tench",
        1: "goldfish",
        2: "great white shark",
        3: "tiger shark",
        4: "hammerhead shark",
        # ... many more classes ...
        151: "Chihuahua",
        152: "Japanese spaniel",
        153: "Maltese dog",
        207: "golden retriever",
        281: "tabby cat",
        282: "tiger cat",
        283: "Persian cat",
        284: "Siamese cat",
        285: "Egyptian cat",
        # ... many more classes ...
    }
    
    return imagenet_classes.get(class_idx, f"Class {class_idx}")


def main():
    """Main function for GradCAM demonstration."""
    args = parse_arguments()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_name = args.model_name
    print(f"Loading model: {model_name}")
    
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Set model to evaluation mode
    model = model.to(device).eval()
    
    # Load and preprocess image
    print(f"Loading image: {args.image_path}")
    original_img, input_tensor = load_image(args.image_path)
    input_tensor = input_tensor.to(device)
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    # Use target class if specified, otherwise use predicted class
    target_class = args.target_class if args.target_class is not None else predicted_class
    class_name = get_imagenet_class_name(target_class)
    print(f"Target class: {target_class} ({class_name})")
    
    # Find target layer for GradCAM
    target_layer = find_target_layer(model, model_type=model_name)
    print(f"Using target layer: {target_layer}")
    
    # Initialize GradCAM method
    method = args.method.lower()
    print(f"Using method: {method}")
    
    if method == "gradcam":
        cam = GradCAM(model, target_layer, device)
    elif method == "gradcam++":
        cam = GradCAMPlusPlus(model, target_layer, device)
    elif method == "smoothgrad":
        cam = SmoothGradCAM(model, target_layer, device, num_samples=25)
    elif method == "xgradcam":
        cam = XGradCAM(model, target_layer, device)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Generate GradCAM heatmap
    heatmap = cam.generate(input_tensor, target_class=target_class)
    
    # Convert heatmap to numpy array and resize to match original image
    heatmap_np = heatmap.squeeze().cpu().numpy()
    
    # Visualize the results
    fig = visualize_gradcam(
        original_img, 
        heatmap_np, 
        alpha=0.5,
        figsize=(12, 4)
    )
    
    # Add title
    fig.suptitle(
        f"Class: {class_name} (Index: {target_class})", 
        fontsize=14,
        y=1.05
    )
    
    # Save results if output path is specified
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        fig.savefig(args.output_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {args.output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
