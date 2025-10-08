# GradientSHAP for WavLM Speech Models

This module provides a comprehensive implementation of GradientSHAP for explaining predictions from WavLM speech models using audio data. The implementation is designed to work with librosa datasets and provides both frame-level and neuron-level explanations.

## Overview

GradientSHAP is an efficient approximation of SHAP values that combines concepts from Integrated Gradients and SHAP. It explains model predictions by:

1. Creating reference samples by interpolating between the input and a random noise distribution
2. Computing gradients for each reference sample with respect to the target class
3. Weighting these gradients according to SHAP principles

This implementation provides explanations that show which parts of an audio file (which frames) most influence the model's prediction.

## Features

- **Frame-level explanations**: Understand which parts of the audio most influence predictions
- **Neuron-level attributions**: Dive deeper to analyze the contribution of individual neurons
- **Easy integration with librosa**: Works with standard audio processing workflows
- **Flexible visualization**: Generate insightful visualizations of attributions over time
- **Efficient batch processing**: Process reference samples in batches for better performance

## Installation Requirements

```bash
pip install torch torchaudio transformers numpy matplotlib seaborn librosa tqdm
```

## Step-by-Step Usage Guide

### 1. Initialize the Explainer

```python
from explainability.LIME_SHAP.ShapeExplainer import GradientSHAP

# Initialize with default parameters
explainer = GradientSHAP(
    model_name='microsoft/wavlm-base-plus',  # WavLM model to explain
    device='cuda',                           # 'cuda' or 'cpu'
    num_samples=50,                          # Number of reference samples
    batch_size=8                             # Batch size for processing
)
```

### 2. Explain an Audio File

```python
# Load audio with librosa
import librosa
waveform, sr = librosa.load('path/to/audio.wav', sr=16000)

# Generate explanations
explanations = explainer.explain(
    torch.tensor(waveform, dtype=torch.float32),
    target_class=2,                # Target emotion class (e.g., 2 for 'happy')
    mode='frame',                  # 'frame' or 'neuron' level explanations
    aggregate_frames='mean'        # How to aggregate frame attributions
)

# Access the attributions
attributions = explanations['attributions']
prediction = explanations['prediction']
print(f"Predicted class: {prediction['class']} with probability {prediction['probability']:.4f}")
```

### 3. Visualize Explanations

```python
# Visualize explanations for an audio file
explainer.visualize_explanations(
    audio_path='path/to/audio.wav',
    target_class=2,                # Target emotion class
    save_path='explanation.png',   # Path to save visualization
    top_k_frames=20                # Number of top frames to highlight
)
```

### 4. Using the Demo Class

For a complete workflow demonstration:

```python
from explainability.LIME_SHAP.ShapeExplainer import WavLMGradientSHAPDemo

# Initialize the demo
demo = WavLMGradientSHAPDemo(
    model_name='microsoft/wavlm-base-plus',
    data_path='path/to/audio/dataset',  # Directory with audio files
    num_classes=8,                      # Number of target classes (e.g., emotions)
    batch_size=4,                       # Batch size for processing
    max_samples=100                     # Maximum number of samples to use
)

# Run the demo
results = demo.run_demo(
    audio_file='path/to/specific/audio.wav',  # Optional: specific file to explain
    target_class=2,                           # Optional: target class
    save_dir='./results'                      # Directory to save visualizations
)
```

## Understanding the Results

The explanations include:

- **attributions**: Numeric attribution values for each frame (or neuron)
- **prediction**: The predicted class and probability
- **mode**: The mode used for explanations ('frame' or 'neuron')

For frame-level explanations, higher attribution values indicate frames that more strongly influence the model's prediction for the target class.

## How It Works

### Reference Sample Generation

The implementation creates reference samples by interpolating between the input and random noise:

```python
# Create random noise with same statistics as input
reference = torch.normal(mean=input_mean, std=input_std, size=(num_samples, input_length))

# Interpolate between reference and input
alphas = torch.linspace(0, 1, num_samples)
samples = reference * (1 - alphas) + input_values * alphas
```

### Attribution Computation

Gradients are computed for each reference sample and integrated according to SHAP principles:

```python
# Compute gradients for each sample
gradients = compute_sample_gradients(samples, target_class)

# Compute SHAP values
shap_values = gradients * input_values
```

### Visualization

The visualization shows:
1. The original audio waveform
2. The spectrogram for frequency analysis
3. The frame-wise attribution values with top frames highlighted

## Example Output

When you run the visualize_explanations method, you'll get a plot showing:

```
+----------------------------------+
|        Audio Waveform            |
+----------------------------------+
|        Spectrogram               |
+----------------------------------+
| Frame-wise GradientSHAP          |
| Attribution                      |
+----------------------------------+
```

## Advanced Usage

### Neuron-Level Explanations

```python
# Initialize with specific feature layer
explainer = GradientSHAP(
    model_name='microsoft/wavlm-base-plus',
    feature_layer=6  # Analyze specific layer
)

# Get neuron-level explanations
explanations = explainer.explain(
    waveform_tensor,
    target_class=2,
    mode='neuron'  # Get neuron-level attributions
)

# Access neuron attributions for a specific layer
layer_attributions = explanations['attributions']['layer_6']
```

### Custom Classifier

```python
# Create a custom classifier for the WavLM model
classifier = torch.nn.Sequential(
    torch.nn.Linear(768, 256),  # WavLM hidden size is 768
    torch.nn.ReLU(),
    torch.nn.Linear(256, num_classes)
)

# Load pre-trained weights if available
classifier.load_state_dict(torch.load('classifier_weights.pth'))

# Initialize explainer with custom classifier
explainer = GradientSHAP(
    model_name='microsoft/wavlm-base-plus',
    classifier=classifier
)
```

## Notes

- The implementation requires sufficient GPU memory to process reference samples
- For large audio files, consider reducing the number of reference samples or processing in smaller chunks
- The accuracy of explanations depends on the quality of the classifier used with WavLM