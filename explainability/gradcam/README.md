# GradCAM Visualization Module

A comprehensive PyTorch implementation of GradCAM and its variants for visualizing and understanding what convolutional neural networks learn.

## Overview

Gradient-weighted Class Activation Mapping (GradCAM) is a technique for producing "visual explanations" for decisions from CNN-based models. It uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.

This package implements:

- **GradCAM**: The original GradCAM algorithm [(Selvaraju et al.)](https://arxiv.org/abs/1610.02391)
- **GradCAM++**: An improved version of GradCAM [(Chattopadhyay et al.)](https://arxiv.org/abs/1710.11063) 
- **Smooth GradCAM**: GradCAM with SmoothGrad technique [(Smilkov et al.)](https://arxiv.org/abs/1706.03825)
- **XGradCAM**: A variant with different weight calculation [(Fu et al.)](https://arxiv.org/abs/2008.02312)

## Installation

### Requirements

- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- NumPy
- Matplotlib
- OpenCV (cv2)
- PIL (Pillow)

### Install Dependencies

```bash
pip install torch torchvision numpy matplotlib opencv-python pillow
```

## Usage

### Basic Example

```python
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from explainability.gradcam import GradCAM, visualize_gradcam, find_target_layer

# Load a pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Find the target layer
target_layer = find_target_layer(model, model_type="resnet")

# Initialize GradCAM
gradcam = GradCAM(model=model, target_layer=target_layer)

# Prepare an image
img = Image.open("cat.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Generate heatmap
heatmap = gradcam.generate(input_tensor)

# Visualize
fig = visualize_gradcam(img, heatmap.squeeze().cpu().numpy())
fig.savefig("gradcam_result.png")
```

### Using Different Variants

```python
from explainability.gradcam import GradCAMPlusPlus, SmoothGradCAM, XGradCAM

# Using GradCAM++
gradcam_pp = GradCAMPlusPlus(model=model, target_layer=target_layer)
heatmap_pp = gradcam_pp.generate(input_tensor)

# Using Smooth GradCAM
smooth_gradcam = SmoothGradCAM(model=model, target_layer=target_layer, num_samples=25)
heatmap_smooth = smooth_gradcam.generate(input_tensor)

# Using XGradCAM
xgradcam = XGradCAM(model=model, target_layer=target_layer)
heatmap_x = xgradcam.generate(input_tensor)
```

## Running the Example Script

An example script is included to demonstrate the usage:

```bash
python example.py --image_path cat.jpg --model_name resnet50 --method gradcam --output_path output.png
```

Arguments:
- `--image_path`: Path to the input image
- `--model_name`: Model architecture to use (resnet18, resnet50, vgg16, densenet121, mobilenet_v2)
- `--method`: GradCAM method to use (gradcam, gradcam++, smoothgrad, xgradcam)
- `--target_class`: Target class index (None for predicted class)
- `--output_path`: Path to save visualization results
- `--use_cuda`: Use CUDA if available


## License

MIT
