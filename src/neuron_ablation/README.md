# Neuron Ablation Framework

A modular framework for performing neuron ablation analysis on transformer-based speech models.

## Overview

This package provides tools for identifying and analyzing important neurons in speech models through various ablation techniques. It supports multiple attribution methods, feature extraction, and visualization tools to help understand the relationship between neurons and speech characteristics.

## Features

- **Multiple Attribution Methods**:
  - Integrated Gradients
  - Direct Gradient
  - Ablation-based Attribution
  - Activation-based Attribution

- **Speech Feature Extraction**:
  - RMS Energy
  - Amplitude Envelope
  - Zero-crossing Rate
  - Spectral Centroid
  - Pitch (F0)
  - Speaker Similarity

- **Layer-wise Ablation Analysis**:
  - Structured analysis across model layers
  - Support for different ablation modes (zero, mean, noise, invert)
  - Temporal spread analysis

- **Advanced Visualization**:
  - Feature change plots
  - Temporal spread heatmaps
  - Emotion probability tracking
  - Upstream norm visualization

## Installation

```bash
# From the repository root
cd src/neuron_ablation
pip install -e .
```

## Usage Example

```python
import torch
from transformers import AutoModel
from neuron_ablation.core import NeuronAblator
from neuron_ablation.visualization import ResultsVisualizer

# Load a speech model
model = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
model.eval()

# Define layer groups for structured analysis
layer_groups = {
    'input': [0],
    'encoder': list(range(1, len(model.encoder.layers))),
    'output': [len(model.encoder.layers)]
}

# Initialize the ablator
ablator = NeuronAblator(
    model=model,
    save_dir="./ablation_results",
    layer_groups=layer_groups
)

# Prepare inputs (audio waveforms)
inputs = torch.randn(4, 16000)  # 4 samples of 1-second audio at 16kHz
labels = torch.tensor([0, 1, 0, 2])  # Example emotion labels

# Run ablation with integrated gradients attribution
results = ablator.run_layerwise_ablation(
    inputs=inputs,
    labels=labels,
    topk=100,  # Ablate top 100 neurons
    ablate_mode="zero",  # Zero-out neuron activations
    target_class=0,  # Target emotion class
    attribution_method="integrated_gradients"
)

# Generate visualizations
visualizer = ResultsVisualizer(ablator)
visualizer.generate_all_plots(results)
```

## Command-line Interface

```bash
python -m neuron_ablation.example --model_name microsoft/wavlm-base-plus \
                                   --data_path /path/to/dataset \
                                   --target_class 0 \
                                   --attribution_method integrated_gradients \
                                   --topk 100
```

## Architecture

The framework is organized into modular components:

- `core.py`: The main `NeuronAblator` class for managing the ablation process
- `attribution.py`: Implementation of various attribution methods
- `feature_extractor.py`: Speech feature extraction functionality
- `visualization.py`: Result visualization tools

## License

MIT