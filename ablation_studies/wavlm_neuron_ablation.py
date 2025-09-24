#!/usr/bin/env python3

import torch
import torchaudio
import numpy as np
from transformers import WavLMModel, Wav2Vec2Processor
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._utils.attribution import NeuronAttribution, PerturbationAttribution
from captum._utils.common import _verify_select_neuron
from captum._utils.gradient import _forward_layer_eval
from typing import Any, Callable, List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
import matplotlib.pyplot as plt
import seaborn as sns

class WavLMNeuronAblation:
    """
    Neuron-level feature ablation for WavLM model on audio data
    """
    
    def __init__(self, model_name="microsoft/wavlm-base-plus", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize WavLM model and processor
        
        Args:
            model_name: HuggingFace model name for WavLM
            device: Device to run the model on
        """
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Store layer information
        self.num_layers = len(self.model.encoder.layers)
        print(f"WavLM model loaded with {self.num_layers} transformer layers")
    
    def preprocess_audio(self, audio_path, target_sr=16000):
        """
        Load and preprocess audio file for WavLM
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sampling rate
            
        Returns:
            Preprocessed audio tensor
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process with WavLM processor
        inputs = self.processor(
            waveform.squeeze().numpy(), 
            sampling_rate=target_sr, 
            return_tensors="pt"
        )
        
        return inputs["input_values"].to(self.device)
    
    def get_layer_module(self, layer_idx):
        """
        Get specific transformer layer module
        
        Args:
            layer_idx: Index of the transformer layer (0 to num_layers-1)
            
        Returns:
            The transformer layer module
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} exceeds model layers {self.num_layers}")
        
        return self.model.encoder.layers[layer_idx]
    
    def create_neuron_forward_func(self, layer_idx, neuron_selector, attribute_to_input=False):
        """
        Create forward function that extracts specific neuron activations
        
        Args:
            layer_idx: Which transformer layer to analyze
            neuron_selector: Which neuron(s) in that layer to focus on
            attribute_to_input: Whether to use layer input or output
            
        Returns:
            Forward function for neuron ablation
        """
        target_layer = self.get_layer_module(layer_idx)
        
        def neuron_forward_func(input_values):
            with torch.no_grad():
                # Get activations from the specific layer
                def hook_fn(module, input_tensor, output_tensor):
                    if attribute_to_input:
                        return input_tensor[0] if isinstance(input_tensor, tuple) else input_tensor
                    else:
                        return output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                
                # Register hook to capture layer activations
                activations = []
                def capture_activation(module, input_tensor, output_tensor):
                    if attribute_to_input:
                        activations.append(input_tensor[0] if isinstance(input_tensor, tuple) else input_tensor)
                    else:
                        activations.append(output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor)
                
                handle = target_layer.register_forward_hook(capture_activation)
                
                try:
                    # Forward pass
                    _ = self.model(input_values)
                    layer_output = activations[0]
                    
                    # Select specific neuron(s)
                    if callable(neuron_selector):
                        return neuron_selector(layer_output)
                    elif isinstance(neuron_selector, (int, tuple)):
                        return self._select_neuron(layer_output, neuron_selector)
                    else:
                        raise ValueError("Invalid neuron_selector type")
                        
                finally:
                    handle.remove()
        
        return neuron_forward_func
    
    def _select_neuron(self, layer_output, neuron_selector):
        """
        Select specific neuron from layer output
        
        Args:
            layer_output: Tensor of shape [batch, seq_len, hidden_dim]
            neuron_selector: Neuron selection specification
            
        Returns:
            Selected neuron activation
        """
        # For transformer: [batch, seq_len, hidden_dim]
        if isinstance(neuron_selector, int):
            # Select specific hidden dimension across all sequence positions
            return layer_output[:, :, neuron_selector].mean(dim=1)  # Average over sequence
        elif isinstance(neuron_selector, tuple):
            if len(neuron_selector) == 2:
                seq_pos, hidden_dim = neuron_selector
                return layer_output[:, seq_pos, hidden_dim]
            elif len(neuron_selector) == 1:
                hidden_dim = neuron_selector[0]
                return layer_output[:, :, hidden_dim].mean(dim=1)
        
        raise ValueError(f"Unsupported neuron_selector format: {neuron_selector}")
    
    def ablate_neuron(self, 
                     audio_input, 
                     layer_idx, 
                     neuron_selector, 
                     baseline_value=0.0,
                     perturbations_per_eval=1,
                     attribute_to_input=False):
        """
        Perform neuron ablation on audio input
        
        Args:
            audio_input: Preprocessed audio tensor
            layer_idx: Which transformer layer to analyze
            neuron_selector: Which neuron to focus on
            baseline_value: Value to replace features with during ablation
            perturbations_per_eval: Number of perturbations per evaluation
            attribute_to_input: Whether to attribute to layer input vs output
            
        Returns:
            Attribution scores for input features
        """
        # Create neuron-specific forward function
        neuron_forward_func = self.create_neuron_forward_func(
            layer_idx, neuron_selector, attribute_to_input
        )
        
        # Create feature ablation instance
        ablator = FeatureAblation(neuron_forward_func)
        
        # Perform attribution
        attributions = ablator.attribute(
            audio_input,
            baselines=baseline_value,
            perturbations_per_eval=perturbations_per_eval
        )
        
        return attributions
    
    def visualize_attributions(self, attributions, audio_input, save_path=None):
        """
        Visualize attribution results
        
        Args:
            attributions: Attribution scores
            audio_input: Original audio input
            save_path: Path to save the plot
        """
        # Convert to numpy for plotting
        attr_np = attributions.squeeze().detach().cpu().numpy()
        audio_np = audio_input.squeeze().detach().cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Plot original audio
        time_axis = np.linspace(0, len(audio_np) / 16000, len(audio_np))
        ax1.plot(time_axis, audio_np)
        ax1.set_title("Original Audio Waveform")
        ax1.set_ylabel("Amplitude")
        
        # Plot attributions
        time_axis_attr = np.linspace(0, len(attr_np) / 16000, len(attr_np))
        ax2.plot(time_axis_attr, attr_np, color='red')
        ax2.set_title("Feature Attributions (Neuron Importance)")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Attribution Score")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_ravdess_sample(data_path, emotion_label=None):
    """
    Load a sample from RAVDESS dataset
    
    Args:
        data_path: Path to RAVDESS data directory
        emotion_label: Specific emotion to load (optional)
        
    Returns:
        Path to audio file and its emotion label
    """
    import os
    import random
    
    # RAVDESS emotion mapping
    emotion_map = {
        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
        5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
    }
    
    audio_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                # RAVDESS filename format: Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion_id = int(parts[2])
                    if emotion_label is None or emotion_map.get(emotion_id) == emotion_label:
                        audio_files.append((os.path.join(root, file), emotion_map.get(emotion_id, "unknown")))
    
    if not audio_files:
        raise ValueError("No RAVDESS audio files found")
    
    return random.choice(audio_files)


# Example usage function
def example_wavlm_neuron_ablation():
    """
    Example of how to use WavLM neuron ablation
    """
    # Initialize the ablation analyzer
    analyzer = WavLMNeuronAblation()
    
    # Load RAVDESS sample (you need to provide the path)
    # audio_path, emotion = load_ravdess_sample("/path/to/ravdess/data", emotion_label="happy")
    
    # For demo, we'll create a dummy audio path - replace with actual RAVDESS path
    audio_path = "path/to/your/ravdess/audio.wav"  # Replace this!
    
    print(f"Processing audio: {audio_path}")
    
    # Preprocess audio
    audio_input = analyzer.preprocess_audio(audio_path)
    print(f"Audio shape: {audio_input.shape}")
    
    # Analyze specific neuron in layer 6, hidden dimension 256
    layer_idx = 6
    neuron_selector = (128, 256)  # (sequence_position, hidden_dimension)
    
    print(f"Analyzing neuron {neuron_selector} in layer {layer_idx}")
    
    # Perform ablation
    attributions = analyzer.ablate_neuron(
        audio_input=audio_input,
        layer_idx=layer_idx,
        neuron_selector=neuron_selector,
        baseline_value=0.0,
        perturbations_per_eval=4
    )
    
    print(f"Attribution shape: {attributions.shape}")
    
    # Visualize results
    analyzer.visualize_attributions(
        attributions, 
        audio_input,
        save_path="wavlm_neuron_ablation_result.png"
    )
    
    return attributions

if __name__ == "__main__":
    example_wavlm_neuron_ablation()