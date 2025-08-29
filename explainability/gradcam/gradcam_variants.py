"""
Extended GradCAM implementations including GradCAM++, Smooth GradCAM, and XGradCAM.

This module provides enhanced versions of GradCAM for better localization and visualization.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
from .gradcam import GradCAM


class GradCAMPlusPlus(GradCAM):
    """
    GradCAM++ implementation, an extension of GradCAM with better localization properties.
    
    Reference: https://arxiv.org/abs/1710.11063
    """
    
    def generate(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[Union[int, torch.Tensor]] = None,
        relu_activations: bool = True
    ) -> torch.Tensor:
        """
        Generate GradCAM++ activation map.
        
        Args:
            input_tensor: Input image tensor of shape (B, C, H, W)
            target_class: Class index for which to generate GradCAM++.
                          If None, the class with the highest score will be used.
            relu_activations: Whether to apply ReLU to the weighted activation map
            
        Returns:
            GradCAM++ heatmap tensor of shape (B, 1, H', W')
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
        gradients = self.gradients[0]  # [B, C, H, W]
        activations = self.activations[0]  # [B, C, H, W]
        
        # GradCAM++ weights calculation
        alpha_numerator = gradients.pow(2)
        alpha_denominator = 2 * gradients.pow(2)
        alpha_denominator += activations * gradients.pow(3)
        alpha_denominator = torch.where(
            alpha_denominator != 0,
            alpha_denominator,
            torch.ones_like(alpha_denominator)
        )
        alpha = alpha_numerator / alpha_denominator
        
        # Weight the activations by the gradients
        weights = torch.sum(F.relu(gradients) * alpha, dim=[2, 3], keepdim=True)
        
        # Apply weights to activations
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


class SmoothGradCAM(GradCAM):
    """
    Smooth GradCAM implementation, which adds noise to the input to generate
    more robust and smoother heatmaps.
    
    Reference: https://arxiv.org/abs/1706.03825
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        target_layer: torch.nn.Module,
        device: Optional[torch.device] = None,
        num_samples: int = 10,
        noise_level: float = 0.1
    ):
        """
        Initialize SmoothGradCAM with a model and target layer.
        
        Args:
            model: PyTorch model to explain
            target_layer: Layer to compute GradCAM on (typically the last convolutional layer)
            device: Device to run computation on (defaults to model's device)
            num_samples: Number of noisy samples to use
            noise_level: Standard deviation of the noise to add to input
        """
        super().__init__(model, target_layer, device)
        self.num_samples = num_samples
        self.noise_level = noise_level
    
    def generate(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[Union[int, torch.Tensor]] = None,
        relu_activations: bool = True
    ) -> torch.Tensor:
        """
        Generate Smooth GradCAM activation map.
        
        Args:
            input_tensor: Input image tensor of shape (B, C, H, W)
            target_class: Class index for which to generate GradCAM.
                          If None, the class with the highest score will be used.
            relu_activations: Whether to apply ReLU to the weighted activation map
            
        Returns:
            Smooth GradCAM heatmap tensor of shape (B, 1, H', W')
        """
        # Ensure input is on the correct device
        input_tensor = input_tensor.to(self.device)
        
        # Get input shape
        batch_size, channels, height, width = input_tensor.shape
        
        # Initialize accumulator for gradients
        accumulated_cams = torch.zeros(batch_size, 1, height, width, device=self.device)
        
        for _ in range(self.num_samples):
            # Add noise to the input tensor
            noise = torch.randn_like(input_tensor) * self.noise_level
            noisy_input = input_tensor + noise
            
            # Reset gradients
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(noisy_input)
            
            # If target_class is not specified, get the index of the maximum score
            if target_class is None:
                current_target_class = torch.argmax(output, dim=1)
            else:
                current_target_class = target_class
            
            # Create a one-hot tensor for the target class
            if isinstance(current_target_class, int):
                target = torch.zeros_like(output)
                target[:, current_target_class] = 1
            else:
                # Handle batch with different target classes
                target = torch.zeros_like(output)
                for i, cls in enumerate(current_target_class):
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
            
            # Add to accumulator
            accumulated_cams += cam
        
        # Average the accumulated GradCAMs
        smooth_cam = accumulated_cams / self.num_samples
        
        # Normalize for visualization
        for i in range(batch_size):
            smooth_cam[i] = smooth_cam[i] - smooth_cam[i].min()
            cam_max = smooth_cam[i].max()
            if cam_max != 0:
                smooth_cam[i] = smooth_cam[i] / cam_max
        
        return smooth_cam


class XGradCAM(GradCAM):
    """
    XGradCAM implementation, which uses a different weighting strategy.
    
    Reference: https://arxiv.org/abs/2008.02312
    """
    
    def generate(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[Union[int, torch.Tensor]] = None,
        relu_activations: bool = True
    ) -> torch.Tensor:
        """
        Generate XGradCAM activation map.
        
        Args:
            input_tensor: Input image tensor of shape (B, C, H, W)
            target_class: Class index for which to generate XGradCAM.
                          If None, the class with the highest score will be used.
            relu_activations: Whether to apply ReLU to the weighted activation map
            
        Returns:
            XGradCAM heatmap tensor of shape (B, 1, H', W')
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
        gradients = self.gradients[0]  # [B, C, H, W]
        activations = self.activations[0]  # [B, C, H, W]
        
        # XGradCAM weights calculation
        weights = torch.sum(gradients * activations, dim=[2, 3], keepdim=True)
        
        # Apply weights to activations
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
