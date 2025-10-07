# Layer-wise Neuron Ablation for WavLM

This directory contains tools for analyzing the relationship between neural activations and acoustic features in the WavLM speech model using ablation analysis techniques.

## Overview

The `LayerwiseLowLevelAblation` class implements a complete workflow for:

1. **Layer-wise Activation Extraction**: Extract activations from each layer of the WavLM model
2. **Attribution Analysis**: Identify important neurons using gradient×activation attribution
3. **Sign-flip Ablation**: Apply perturbations to important neurons by flipping their sign
4. **Feature Proxy Computation**: Compute activation-space proxies for acoustic features
5. **Visualization**: Analyze how feature proxies change when important neurons are ablated

## Low-level Features

The implementation focuses on three key low-level acoustic features:

- **RMS Energy**: Captured through L2 norm across hidden dimensions
- **Zero-Crossing Rate**: Approximated by zero-crossings in the first PCA component
- **Spectral Centroid**: Estimated from spectral analysis of the RMS proxy

## Usage

```python
# Initialize the ablator
ablator = LayerwiseLowLevelAblation(
    model_name='microsoft/wavlm-base-plus',
    data_path="/path/to/RAVDESS_dataset",
    save_dir="./wavlm_ablation_results",
    max_samples=200
)

# Run analysis for 'happy' emotion (class index 2 in RAVDESS)
results, fig_path = ablator.run_complete_analysis(
    target_class=2,
    topk=50,
    frame_policy='max_frame',
    num_batches=5
)
```

## Kaggle Compatibility

The implementation is optimized for Kaggle notebooks:

- Automatic path detection for RAVDESS dataset
- Memory-efficient batch processing
- Error handling for Kaggle environment
- Integration with IPython display for in-notebook visualization

## References

- WavLM: https://huggingface.co/microsoft/wavlm-base-plus
- RAVDESS Dataset: https://zenodo.org/record/1188976