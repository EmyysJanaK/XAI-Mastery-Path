"""
Custom class for layer-wise neuron ablation analysis on WavLM
focusing specifically on low-level acoustic feature representations.
"""

from LayerwiseLowLevelAblation import LayerwiseLowLevelAblation

# Instantiate the ablator for Kaggle environment
import os

# Configure paths for Kaggle
data_path = "/kaggle/input/ravdess-emotional-speech-audio"
save_dir = "/kaggle/working/wavlm_ablation_results"

# Alternative path format for some Kaggle environments
if not os.path.exists(data_path):
    data_path = "../input/ravdess-emotional-speech-audio"

# Create save directory
os.makedirs(save_dir, exist_ok=True)

# Initialize the ablator with appropriate parameters for Kaggle
try:
    ablator = LayerwiseLowLevelAblation(
        model_name='microsoft/wavlm-base-plus',
        data_path=data_path,
        save_dir=save_dir,
        max_samples=100,  # Reduced sample size for Kaggle
        batch_size=4      # Smaller batch size to avoid OOM errors
    )
    
    print("LayerwiseLowLevelAblation initialized successfully!")
    print(f"Data path: {data_path}")
    print(f"Save directory: {save_dir}")
    print("Ready to run analysis. Use ablator.run_complete_analysis() to begin.")
    
except Exception as e:
    print(f"Error during initialization: {e}")
    import traceback
    traceback.print_exc()

# Example usage (commented out to allow manual execution)
'''
# Run analysis for 'happy' emotion (class index 2 in RAVDESS)
results, fig_path = ablator.run_complete_analysis(
    target_class=2,    # Happy emotion
    topk=50,           # Analyze top 50 neurons per layer
    frame_policy='max_frame',  # Ablate only at max activation frame
    num_batches=3      # Process 3 batches (adjust based on your needs)
)

# Display the results
from IPython.display import Image, display
display(Image(filename=fig_path))
'''