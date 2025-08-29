"""
GradCAM (Gradient-weighted Class Activation Mapping) Implementation.

This module provides tools to visualize and understand what CNNs learn and how they make decisions.
GradCAM uses the gradients of any target concept flowing into the final convolutional layer
to produce a coarse localization map highlighting important regions in the image for predicting the concept.

Reference paper: https://arxiv.org/abs/1610.02391
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np


class GradCAM:
    """
    GradCAM implementation for visualizing CNN decisions.
    
    Attributes:
        model (torch.nn.Module): The model to explain
        target_layer (torch.nn.Module): The target layer for visualization
        device (torch.device): Device to run computation on
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        target_layer: torch.nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Initialize GradCAM with a model and target layer.
        
        Args:
            model: PyTorch model to explain
            target_layer: Layer to compute GradCAM on (typically the last convolutional layer)
            device: Device to run computation on (defaults to model's device)
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device or next(model.parameters()).device
        
        self.gradients: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []
        
        # Register hooks
        self._register_hooks()
        
        # Set model to evaluation mode
        self.model.eval()
        
    def _register_hooks(self) -> None:
        """Register forward and backward hooks to the target layer."""
        
        # Hook for storing activations
        def forward_hook(module, input, output):
            self.activations = [output.detach()]
        
        # Hook for storing gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = [grad_output[0].detach()]
        
        # Register the hooks
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def _release_hooks(self) -> None:
        """Remove the registered hooks."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._release_hooks()
    
    def generate(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[Union[int, torch.Tensor]] = None,
        relu_activations: bool = True
    ) -> torch.Tensor:
        """
        Generate GradCAM activation map.
        
        Args:
            input_tensor: Input image tensor of shape (B, C, H, W)
            target_class: Class index for which to generate GradCAM.
                          If None, the class with the highest score will be used.
            relu_activations: Whether to apply ReLU to the weighted activation map
            
        Returns:
            GradCAM heatmap tensor of shape (B, 1, H', W')
        """
        # Ensure input is on the correct device
        input_tensor = input_tensor.to(self.device)
        
        # Reset gradients
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If target_class is not specified, get the index of the maximum score
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # Create a one-hot tensor for the target class
        if isinstance(target_class, int):
            target = torch.zeros_like(output)
            target[:, target_class] = 1
        else:
            # Handle batch with different target classes
            target = torch.zeros_like(output)
            for i, cls in enumerate(target_class):
                target[i, cls] = 1
        
        # Backward pass
        output.backward(gradient=target, retain_graph=True)
        
        # Get activations and gradients
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weight the activations by the gradients
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU if specified
        if relu_activations:
            cam = F.relu(cam)
        
        # Normalize for visualization
        batch_size = cam.shape[0]
        for i in range(batch_size):
            cam[i] = cam[i] - cam[i].min()
            cam_max = cam[i].max()
            if cam_max != 0:
                cam[i] = cam[i] / cam_max
        
        return cam
    
    def __call__(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[Union[int, torch.Tensor]] = None,
        relu_activations: bool = True
    ) -> torch.Tensor:
        """
        Alias for generate method.
        
        Args:
            input_tensor: Input image tensor of shape (B, C, H, W)
            target_class: Class index for which to generate GradCAM.
                          If None, the class with the highest score will be used.
            relu_activations: Whether to apply ReLU to the weighted activation map
            
        Returns:
            GradCAM heatmap tensor of shape (B, 1, H', W')
        """
        return self.generate(input_tensor, target_class, relu_activations)
