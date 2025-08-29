"""
Visualization utilities for GradCAM.

This module provides functions to visualize GradCAM heatmaps on images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, Union, List
from PIL import Image
import cv2


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor to the range [0, 1].
    
    Args:
        tensor: Input tensor to normalize
        
    Returns:
        Normalized tensor
    """
    tensor = tensor - tensor.min()
    if tensor.max() > 0:
        tensor = tensor / tensor.max()
    return tensor


def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Apply a colormap to a grayscale heatmap.
    
    Args:
        heatmap: Grayscale heatmap as numpy array with values in [0, 1]
        colormap: OpenCV colormap to apply
        
    Returns:
        Colored heatmap as RGB numpy array with values in [0, 255]
    """
    # Scale to [0, 255] and convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert BGR to RGB
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    return colored_heatmap


def overlay_heatmap(
    image: np.ndarray, 
    heatmap: np.ndarray, 
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay a heatmap on an image.
    
    Args:
        image: Original image as numpy array with values in [0, 255]
        heatmap: Colored heatmap as numpy array with values in [0, 255]
        alpha: Transparency factor for the overlay
        
    Returns:
        Overlaid image as numpy array with values in [0, 255]
    """
    # Ensure image is RGB
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA image
        image = image[:, :, :3]
    
    # Resize heatmap to match image size if needed
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Overlay heatmap on image
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlaid


def visualize_gradcam(
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    heatmap: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> Figure:
    """
    Visualize an image with its GradCAM heatmap overlay.
    
    Args:
        image: Original image (numpy array, tensor, or PIL image)
        heatmap: GradCAM heatmap (numpy array or tensor)
        alpha: Transparency factor for the overlay
        colormap: OpenCV colormap to apply to the heatmap
        figsize: Figure size
        save_path: Path to save the visualization (if None, no saving)
        
    Returns:
        Matplotlib figure object
    """
    # Convert image to numpy if needed
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
        # Move channel dimension if needed (from [C, H, W] to [H, W, C])
        if image_np.shape[0] in [1, 3, 4]:
            image_np = np.transpose(image_np, (1, 2, 0))
            
        # Convert to uint8 if needed
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
            
    elif isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Convert heatmap to numpy if needed
    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap.cpu().numpy()
        # Squeeze if needed (remove batch and channel dimensions)
        if len(heatmap_np.shape) > 2:
            heatmap_np = np.squeeze(heatmap_np)
    else:
        heatmap_np = heatmap
    
    # Normalize heatmap if needed
    if heatmap_np.max() > 1.0:
        heatmap_np = heatmap_np / 255.0
    
    # Apply colormap to heatmap
    colored_heatmap = apply_colormap(heatmap_np, colormap)
    
    # Overlay heatmap on image
    overlaid = overlay_heatmap(image_np, colored_heatmap, alpha)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot heatmap
    axes[1].imshow(colored_heatmap)
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # Plot overlaid image
    axes[2].imshow(overlaid)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig
