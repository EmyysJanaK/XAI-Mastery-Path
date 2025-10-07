# Layerwise Low-Level Ablation for Speech Models

This repository provides a PyTorch implementation of layer-wise neuron ablation analysis specifically designed for WavLM speech models. It allows researchers to visualize how low-level acoustic feature proxies change when important neuron activations are perturbed through sign-flip ablation.

## Key Features

- **Sign-Flip Ablation**: Instead of zeroing out neurons, this implementation flips their sign (multiplying by -1) to provide a more disruptive yet energy-preserving perturbation
- **Gradient×Activation Attribution**: Directly implements the gradient×activation attribution method in PyTorch without relying on Captum
- **Low-Level Acoustic Feature Proxies**: Computes interpretable proxies for common acoustic features:
  - **RMS Energy Proxy**: L2 norm across hidden dimensions
  - **Zero-Crossing Rate Proxy**: Zero-crossings in the first PCA component
  - **Spectral Centroid Proxy**: Centroid of FFT magnitude spectrum

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/XAI-Mastery-Path.git
cd XAI-Mastery-Path

# Install dependencies
pip install torch torchaudio transformers numpy matplotlib seaborn scikit-learn tqdm
```

## Usage

### Basic Usage

```python
from LayerwiseLowLevelAblation import LayerwiseLowLevelAblation

# Initialize the ablation analyzer
ablator = LayerwiseLowLevelAblation(
    model_name='microsoft/wavlm-base-plus',
    data_path='/path/to/ravdess/dataset',
    save_dir='./results',
    max_samples=200,
    batch_size=4
)

# Run the analysis for the 'happy' emotion (class 2 in RAVDESS)
results, fig_path = ablator.run_complete_analysis(
    target_class=2,
    topk=50,  # Top 50 neurons per layer
    frame_policy='max_frame'  # Ablate only at frame with maximum activation
)

print(f"Analysis complete! Visualization saved to {fig_path}")
```

### Command-Line Interface

For convenience, we provide a command-line script:

```bash
python run_layerwise_analysis.py \
    --ravdess_path /path/to/ravdess \
    --output_dir ./results \
    --emotion happy \
    --samples 200 \
    --batch_size 4 \
    --topk 50 \
    --frame_policy max_frame
```

## The RAVDESS Dataset

This implementation uses the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), which contains speech recordings with various emotions. You can download the dataset from:

[RAVDESS Dataset](https://zenodo.org/record/1188976)

The dataset includes 24 professional actors vocalizing two lexically-matched statements in eight emotions:
- neutral (0)
- calm (1)
- happy (2)
- sad (3)
- angry (4)
- fearful (5)
- disgust (6)
- surprised (7)

## How It Works

### 1. Attribution

The implementation first identifies important neurons for each layer using the gradient×activation attribution method:

1. Run a forward pass to get activations
2. Set up each layer's activations to track gradients
3. Compute logits for the target class
4. Run backward pass to get gradients
5. Multiply gradients with activations to get attribution scores
6. Select top-k neurons based on these scores

### 2. Sign-Flip Ablation

Unlike traditional ablation that zeros out neurons, sign-flip ablation multiplies the activation by -1:

```python
# Traditional zeroing
activation[neuron_idx] = 0

# Sign-flip ablation (this implementation)
activation[neuron_idx] *= -1
```

This approach is more disruptive than zeroing while preserving energy, helping identify neurons that encode specific directional information.

### 3. Feature Proxy Computation

The implementation computes three interpretable proxies for common acoustic features:

1. **RMS Energy Proxy**: L2 norm across hidden dimensions of activations
2. **Zero-Crossing Rate Proxy**: Zero-crossings in the first PCA component of activations
3. **Spectral Centroid Proxy**: Centroid of FFT magnitude spectrum of the RMS proxy

### 4. Visualization

The results are visualized as normalized changes in each feature proxy across layers, with standard error bands.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.