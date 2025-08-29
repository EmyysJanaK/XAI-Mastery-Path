"""
Utility functions for finding and selecting model layers for GradCAM.

This module provides helper functions to identify appropriate layers
in different model architectures for GradCAM visualization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import re


def find_layer_types_in_model(
    model: nn.Module,
    layer_types: List[type]
) -> Dict[str, nn.Module]:
    """
    Find all layers of specific types in a model.
    
    Args:
        model: PyTorch model to search
        layer_types: List of layer types to find
        
    Returns:
        Dictionary mapping layer names to layer modules
    """
    result = {}
    for name, module in model.named_modules():
        if any(isinstance(module, layer_type) for layer_type in layer_types):
            result[name] = module
    return result


def find_conv_layers(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Find all convolutional layers in a model.
    
    Args:
        model: PyTorch model to search
        
    Returns:
        Dictionary mapping layer names to convolutional layers
    """
    conv_types = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
    return find_layer_types_in_model(model, conv_types)


def find_last_conv_layer(model: nn.Module) -> Tuple[str, nn.Module]:
    """
    Find the last convolutional layer in a model.
    
    Args:
        model: PyTorch model to search
        
    Returns:
        Tuple of (layer_name, layer_module) for the last convolutional layer
    
    Raises:
        ValueError: If no convolutional layer is found
    """
    conv_layers = find_conv_layers(model)
    if not conv_layers:
        raise ValueError("No convolutional layer found in the model")
    
    # Get the last layer by name (typically, deeper layers have longer names)
    last_conv_name = sorted(conv_layers.keys())[-1]
    return last_conv_name, conv_layers[last_conv_name]


def get_target_layer_by_name(
    model: nn.Module, 
    layer_name: str
) -> nn.Module:
    """
    Get a specific layer from a model by name.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer to retrieve
        
    Returns:
        The requested layer module
        
    Raises:
        ValueError: If the layer is not found
    """
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"Layer '{layer_name}' not found in the model")


def find_target_layer(
    model: nn.Module,
    target_layer_name: Optional[str] = None,
    model_type: Optional[str] = None
) -> nn.Module:
    """
    Find a target layer for GradCAM based on model type or specified layer name.
    
    Args:
        model: PyTorch model
        target_layer_name: Name of the target layer (if specified)
        model_type: Type of the model architecture (e.g., 'resnet', 'vgg', etc.)
        
    Returns:
        Target layer module for GradCAM visualization
        
    Raises:
        ValueError: If the target layer cannot be determined
    """
    # If target layer is specified by name, try to find it
    if target_layer_name is not None:
        try:
            return get_target_layer_by_name(model, target_layer_name)
        except ValueError:
            pass  # Fall back to automatic detection
    
    # If model_type is specified, use architecture-specific heuristics
    if model_type is not None:
        model_type = model_type.lower()
        
        if 'resnet' in model_type:
            # For ResNet, use the last layer in the last bottleneck/block
            if hasattr(model, 'layer4'):
                return model.layer4[-1].conv3 if hasattr(model.layer4[-1], 'conv3') else model.layer4[-1].conv2
        
        elif 'vgg' in model_type:
            # For VGG, use the last convolutional layer
            if hasattr(model, 'features'):
                conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
                return conv_layers[-1]
        
        elif 'densenet' in model_type:
            # For DenseNet, use the last dense block's transition layer
            if hasattr(model, 'features'):
                return model.features.norm5
        
        elif 'mobilenet' in model_type:
            # For MobileNet, use the last convolutional layer
            if hasattr(model, 'features'):
                return model.features[-1]
        
        elif 'efficientnet' in model_type:
            # For EfficientNet, use the last convolutional layer
            if hasattr(model, 'features'):
                conv_layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
                return conv_layers[-1]
    
    # If we reach here, try to find the last convolutional layer
    try:
        _, layer = find_last_conv_layer(model)
        return layer
    except ValueError:
        raise ValueError(
            "Could not automatically determine target layer. "
            "Please specify target_layer_name or model_type."
        )
