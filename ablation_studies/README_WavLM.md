# WavLM Neuron Ablation with RAVDESS Dataset

This guide explains how to perform neuron-level ablation analysis on WavLM using the RAVDESS emotion dataset.

## Prerequisites

### 1. Download RAVDESS Dataset
- Go to: https://zenodo.org/record/1188976
- Download "Audio-only files" (about 1.98 GB)
- Extract to a folder, you'll get structure like:
  ```
  ravdess-emotional-speech-audio/
    Actor_01/
      03-01-01-01-01-01-01.wav
      03-01-01-01-01-02-01.wav
      ...
    Actor_02/
      ...
  ```

### 2. Install Dependencies
```bash
pip install torch torchaudio transformers captum datasets librosa numpy matplotlib seaborn
```

## Quick Start

### Step 1: Update Paths
Edit `wavlm_ravdess_integration.py` and set your RAVDESS path:
```python
RAVDESS_PATH = "/path/to/your/ravdess-emotional-speech-audio/"
```

### Step 2: Run Basic Analysis
```bash
python wavlm_ravdess_integration.py
```

### Step 3: Command Line Usage
```bash
# Analyze specific emotion and neuron
python wavlm_ravdess_integration.py --ravdess_path /path/to/ravdess --emotion happy --layer 8 --neuron 256

# Compare multiple emotions
python wavlm_ravdess_integration.py --ravdess_path /path/to/ravdess --compare --layer 6 --neuron 384
```

## Understanding the Results

### What the Analysis Shows
1. **Original Audio**: The input speech waveform
2. **Attribution Scores**: How much each audio time segment contributes to the specific neuron's activation
3. **Positive values**: Audio segments that increase neuron activation
4. **Negative values**: Audio segments that decrease neuron activation

### Interpretation Examples
- **High attribution in vocal regions**: The neuron responds strongly to speech content
- **High attribution in silence**: The neuron might be detecting speech boundaries
- **Emotion-specific patterns**: Different emotions may show different attribution patterns

## Advanced Usage

### 1. Analyze Different Layers
```python
# Compare early vs late layers
for layer in [2, 6, 10]:
    run_wavlm_neuron_analysis(
        ravdess_path=RAVDESS_PATH,
        emotion="happy",
        layer_idx=layer,
        neuron_dim=256
    )
```

### 2. Analyze Multiple Neurons
```python
# Compare different neurons in same layer
for neuron in [128, 256, 384, 512]:
    run_wavlm_neuron_analysis(
        ravdess_path=RAVDESS_PATH,
        emotion="angry",
        layer_idx=8,
        neuron_dim=neuron
    )
```

### 3. Gender-Specific Analysis
```python
from ravdess_handler import RAVDESSDataHandler

handler = RAVDESSDataHandler(RAVDESS_PATH)

# Male speakers only
male_sample = handler.get_random_sample(emotion="sad", gender="male")

# Female speakers only  
female_sample = handler.get_random_sample(emotion="sad", gender="female")
```

## File Structure
```
ablation_studies/
├── NeuronFeatureAblation.py          # Original Captum implementation
├── wavlm_neuron_ablation.py          # WavLM-specific ablation class
├── ravdess_handler.py                # RAVDESS dataset handler
├── wavlm_ravdess_integration.py      # Complete integration script
├── requirements_wavlm.txt            # Dependencies
└── results/                          # Output directory
    ├── plots/
    └── data/
```

## Research Questions You Can Explore

1. **Layer Specialization**: Do different layers focus on different aspects (phonetics vs emotions)?
2. **Emotion Encoding**: Which neurons are most sensitive to specific emotions?
3. **Temporal Patterns**: Do neurons respond to specific temporal patterns in speech?
4. **Gender Differences**: Are there neurons that respond differently to male vs female voices?
5. **Intensity Effects**: How do normal vs strong emotional intensity affect neuron responses?

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce `perturbations_per_eval` parameter
2. **File Not Found**: Check RAVDESS path and file structure
3. **Import Errors**: Ensure all dependencies are installed
4. **Slow Processing**: Use smaller audio segments or reduce perturbation count

### Performance Tips
1. Use GPU if available: `device="cuda"`
2. Increase `perturbations_per_eval` for faster processing (if memory allows)
3. Process shorter audio clips for quicker experimentation

## Expected Output

The analysis will generate:
1. **Visualization plots** showing attribution patterns
2. **Statistical summaries** of neuron responses
3. **Saved results** (.pt files) for further analysis
4. **Comparison plots** across emotions/conditions

## Next Steps

After running basic analysis, you can:
1. Modify the neuron selection strategy
2. Implement custom baseline methods
3. Add statistical significance testing
4. Create interactive visualizations
5. Extend to other speech emotion datasets