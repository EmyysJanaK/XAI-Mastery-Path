"""
GradCAM module for visualizing CNN decisions.

This package provides tools to visualize which parts of an image contribute
most to a particular classification decision in convolutional neural networks.

Core GradCAM implementation is based on the paper:
"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
by Selvaraju et al. (https://arxiv.org/abs/1610.02391)

The module also includes several GradCAM variants:
- GradCAM++: https://arxiv.org/abs/1710.11063
- Smooth GradCAM: Based on SmoothGrad (https://arxiv.org/abs/1706.03825)
- XGradCAM: https://arxiv.org/abs/2008.02312
"""

from .gradcam import GradCAM
from .gradcam_variants import GradCAMPlusPlus, SmoothGradCAM, XGradCAM
from .visualization import visualize_gradcam, apply_colormap, overlay_heatmap
from .layer_utils import find_target_layer, find_conv_layers, get_target_layer_by_name

__all__ = [
    'GradCAM',
    'GradCAMPlusPlus',
    'SmoothGradCAM',
    'XGradCAM',
    'visualize_gradcam',
    'apply_colormap',
    'overlay_heatmap',
    'find_target_layer',
    'find_conv_layers',
    'get_target_layer_by_name',
]

__version__ = '1.0.0'
