"""
ShapeExplainer.py: Implementation of GradientSHAP for WavLM speech models.

This module provides classes for explaining WavLM model predictions on speech data
using the GradientSHAP algorithm. It supports audio data from librosa datasets
and produces frame-wise and neuron-level explanations.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from tqdm import tqdm
import random
from transformers import AutoModel, AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader


class LibrosaAudioDataset(Dataset):
    """
    Dataset class for loading audio files using librosa.
    
    This dataset can load audio files from a directory and prepare them for
    use with WavLM and similar speech models.
    """
    
    def __init__(self, data_path, feature_extractor, labels=None, sample_rate=16000, max_samples=None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to audio files directory
            feature_extractor: WavLM feature extractor for audio preprocessing
            labels (dict, optional): Dictionary mapping filenames to labels
            sample_rate (int): Target sample rate for audio
            max_samples (int, optional): Maximum number of samples to use
            transform (callable, optional): Optional transform to apply to audio
        """
        self.data_path = data_path
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.transform = transform
        self.labels = labels or {}
        
        # List all audio files
        self.audio_files = []
        
        # Walk through the directory
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    self.audio_files.append(os.path.join(root, file))
        
        # Limit dataset size if specified
        if max_samples and max_samples < len(self.audio_files):
            self.audio_files = random.sample(self.audio_files, max_samples)
            
        print(f"Loaded {len(self.audio_files)} audio files from {data_path}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: A dictionary with input_values, label (if available), and filename
        """
        audio_path = self.audio_files[idx]
        filename = os.path.basename(audio_path)
        
        # Load and resample audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Apply transform if specified
        if self.transform:
            waveform = self.transform(waveform)
        
        # Convert to float32 tensor
        waveform = torch.tensor(waveform, dtype=torch.float32)
        
        # Process with feature extractor
        inputs = self.feature_extractor(waveform, sampling_rate=self.sample_rate, return_tensors="pt")
        
        # Get label if available
        label = self.labels.get(filename, -1)
        
        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": label,
            "filename": filename
        }


class GradientSHAP:
    """
    GradientSHAP implementation for WavLM speech models.
    
    This class implements GradientSHAP for speech models by:
    1. Creating reference samples by mixing the target input with random noise
    2. Computing gradients for each reference sample
    3. Weighting the gradients according to SHAP principles
    
    This allows frame-wise and neuron-level attributions to understand which
    parts of the audio input most influence the model's predictions.
    """
    
    def __init__(
        self,
        model_name='microsoft/wavlm-base-plus',
        device=None,
        num_samples=50,
        feature_layer=None,
        classifier=None,
        batch_size=8
    ):
        """
        Initialize the GradientSHAP explainer.
        
        Args:
            model_name (str): HuggingFace model name for WavLM
            device (str): Device to use ('cuda' or 'cpu')
            num_samples (int): Number of reference samples for SHAP
            feature_layer (int, optional): Layer to extract features from (None for last layer)
            classifier (torch.nn.Module, optional): Classification head
            batch_size (int): Batch size for processing reference samples
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Parameters
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.feature_layer = feature_layer
        
        # Set up classifier if not provided
        if classifier is None:
            hidden_size = self.model.config.hidden_size
            self.classifier = torch.nn.Linear(hidden_size, 8)  # Default: 8 emotions
        else:
            self.classifier = classifier
            
        self.classifier.to(self.device)
        
        # For storing activations
        self.activation_store = {}
        self.hooks = []
        
        # Register hooks for specified feature layer
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register forward hooks to capture activations from the model.
        
        If a specific feature layer is specified, only register a hook for that layer.
        Otherwise, register hooks for all encoder layers.
        """
        try:
            def get_activation(name):
                def hook(module, input, output):
                    # Store activations - supports both tuple and tensor outputs
                    if isinstance(output, tuple):
                        self.activation_store[name] = output[0].detach()
                    else:
                        self.activation_store[name] = output.detach()
                return hook
            
            # Register hooks for specific or all encoder layers
            if self.feature_layer is not None:
                # Register hook for just the specified layer
                if 0 <= self.feature_layer < len(self.model.encoder.layers):
                    layer = self.model.encoder.layers[self.feature_layer]
                    h = layer.register_forward_hook(get_activation(self.feature_layer))
                    self.hooks.append(h)
                    print(f"Registered hook for layer {self.feature_layer}")
            else:
                # Register hooks for all encoder layers
                for i, layer in enumerate(self.model.encoder.layers):
                    h = layer.register_forward_hook(get_activation(i))
                    self.hooks.append(h)
                print(f"Registered hooks for all {len(self.hooks)} encoder layers")
                
        except Exception as e:
            print(f"Error registering hooks: {e}")
            raise
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        print("Removed all hooks")
        
    def __del__(self):
        """Clean up by removing hooks when object is deleted"""
        if hasattr(self, 'hooks') and self.hooks:
            self.remove_hooks()
    
    def generate_reference_samples(self, input_values, num_samples=None):
        """
        Generate reference samples by interpolating between input and random noise.
        
        This is a key step in GradientSHAP, where we create samples that lie on the
        path between a reference distribution (random noise) and the actual input.
        
        Args:
            input_values (torch.Tensor): Input audio values [T]
            num_samples (int, optional): Number of samples to generate
            
        Returns:
            torch.Tensor: Reference samples [N, T]
            torch.Tensor: Interpolation coefficients [N, 1]
        """
        if num_samples is None:
            num_samples = self.num_samples
            
        # Create random reference noise with same shape as input
        # Use a normal distribution with same mean and std as input
        input_mean = input_values.mean()
        input_std = input_values.std()
        
        # Create reference distribution (random noise)
        reference = torch.normal(
            mean=input_mean,
            std=input_std,
            size=(num_samples, input_values.shape[0])
        ).to(self.device)
        
        # Create alphas for interpolation
        alphas = torch.linspace(0, 1, num_samples).view(-1, 1).to(self.device)
        
        # Generate samples by interpolating between reference and input
        samples = reference * (1 - alphas) + input_values * alphas
        
        return samples, alphas
    
    def explain(
        self, 
        input_values, 
        target_class=None,
        mode='frame',
        aggregate_frames='mean'
    ):
        """
        Generate GradientSHAP explanations for a given input.
        
        Args:
            input_values (torch.Tensor): Input audio values
            target_class (int, optional): Target class for explanation
            mode (str): 'frame' or 'neuron' level explanations
            aggregate_frames (str): Method to aggregate frame attributions ('mean', 'sum', 'max')
            
        Returns:
            Dict: Explanation results including attributions
        """
        print(f"Generating GradientSHAP explanations in {mode} mode")
        
        # Move input to device
        if not isinstance(input_values, torch.Tensor):
            input_values = torch.tensor(input_values, dtype=torch.float32)
        
        input_values = input_values.to(self.device)
        
        # Generate reference samples
        samples, alphas = self.generate_reference_samples(input_values)
        num_samples = samples.shape[0]
        
        # Process in batches
        all_gradients = []
        
        for batch_idx in tqdm(range(0, num_samples, self.batch_size), desc="Computing gradients"):
            batch_end = min(batch_idx + self.batch_size, num_samples)
            batch_samples = samples[batch_idx:batch_end]
            batch_alphas = alphas[batch_idx:batch_end]
            
            # Get gradients for this batch
            batch_gradients = self._compute_sample_gradients(batch_samples, target_class)
            all_gradients.append(batch_gradients)
        
        # Combine gradients from all batches
        all_gradients = torch.cat(all_gradients, dim=0)
        
        # Compute integrated gradients by weighting according to SHAP formulation
        shap_values = self._compute_shap_values(all_gradients, alphas, input_values, mode, aggregate_frames)
        
        # Run forward pass to get predictions
        with torch.no_grad():
            inputs = self.feature_extractor(input_values, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Get class predictions using classifier
            if hasattr(outputs, 'last_hidden_state'):
                pooled = outputs.last_hidden_state.mean(dim=1)
                logits = self.classifier(pooled)
                probs = torch.softmax(logits, dim=1)
                
                # Get predicted class if target not specified
                if target_class is None:
                    target_class = torch.argmax(probs, dim=1).item()
                
                prediction = {
                    'class': target_class,
                    'probability': probs[0, target_class].item()
                }
            else:
                prediction = {'class': -1, 'probability': 0.0}
            
        return {
            'attributions': shap_values,
            'prediction': prediction,
            'mode': mode,
            'aggregate_method': aggregate_frames,
        }
    
    def _compute_sample_gradients(self, samples, target_class=None):
        """
        Compute gradients for sample inputs with respect to the target class.
        
        Args:
            samples (torch.Tensor): Batch of input samples [B, T]
            target_class (int, optional): Target class for explanation
            
        Returns:
            torch.Tensor: Gradients for each sample
        """
        batch_size = samples.shape[0]
        gradients = []
        
        for i in range(batch_size):
            sample = samples[i:i+1]  # Keep batch dimension [1, T]
            sample.requires_grad_(True)
            
            # Process with feature extractor
            inputs = self.feature_extractor(sample[0], sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            self.model.zero_grad()
            if hasattr(self.classifier, 'zero_grad'):
                self.classifier.zero_grad()
            
            outputs = self.model(**inputs)
            
            # Get predictions from classifier
            if hasattr(outputs, 'last_hidden_state'):
                pooled = outputs.last_hidden_state.mean(dim=1)
                logits = self.classifier(pooled)
                
                # Get target class if not specified
                if target_class is None:
                    target_class = logits.argmax(dim=1).item()
                
                # Get target class logit
                target_logit = logits[0, target_class]
                
                # Backward pass to get gradients
                target_logit.backward()
                
                # Get gradients with respect to input
                grad = sample.grad.detach()
                gradients.append(grad)
            
            # Clean up
            if sample.grad is not None:
                sample.grad.zero_()
        
        # Combine gradients
        return torch.cat(gradients, dim=0)
    
    def _compute_shap_values(self, gradients, alphas, input_values, mode='frame', aggregate_frames='mean'):
        """
        Compute final SHAP values from gradients.
        
        Args:
            gradients (torch.Tensor): Gradients from all samples [N, T]
            alphas (torch.Tensor): Interpolation coefficients [N, 1]
            input_values (torch.Tensor): Original input values [T]
            mode (str): 'frame' or 'neuron' level explanations
            aggregate_frames (str): Method to aggregate frame attributions
            
        Returns:
            torch.Tensor or Dict: SHAP values for the input
        """
        # Apply trapezoidal rule for integration
        shap_values = gradients * input_values
        
        # For neuron-level explanations, we need to access the activation store
        if mode == 'neuron' and self.activation_store:
            # Get activation from the latest forward pass
            # We need to do one more forward pass to capture the activations
            with torch.no_grad():
                inputs = self.feature_extractor(input_values, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                _ = self.model(**inputs)
            
            # Extract neuron-level explanations
            neuron_attributions = {}
            
            for layer_idx, activations in self.activation_store.items():
                # If we are only interested in specific layer
                if self.feature_layer is not None and layer_idx != self.feature_layer:
                    continue
                    
                # Get shape information
                batch_size, seq_len, hidden_dim = activations.shape
                
                # For each neuron, compute importance
                layer_attributions = torch.zeros((seq_len, hidden_dim)).to(self.device)
                
                # We use the integrated gradients to estimate neuron importance
                # This is a simplification - in practice we'd need to compute
                # gradients w.r.t each neuron activation
                for frame_idx in range(seq_len):
                    for neuron_idx in range(hidden_dim):
                        # Estimate neuron importance by corresponding gradient
                        # This is approximate - full GradientSHAP would need to integrate
                        # over neuron activation paths
                        neuron_attr = gradients[:, frame_idx].sum()
                        layer_attributions[frame_idx, neuron_idx] = neuron_attr
                
                neuron_attributions[f"layer_{layer_idx}"] = layer_attributions.cpu().numpy()
            
            return neuron_attributions
        
        # For frame-level explanations, we aggregate across frames
        else:
            # Convert to numpy for easier handling
            frame_attributions = shap_values.cpu().numpy()
            
            # Aggregate if requested
            if aggregate_frames == 'mean':
                return frame_attributions.mean(axis=0)
            elif aggregate_frames == 'sum':
                return frame_attributions.sum(axis=0)
            elif aggregate_frames == 'max':
                return frame_attributions.max(axis=0)
            else:
                return frame_attributions
    
    def visualize_explanations(self, audio_path, target_class=None, save_path=None, top_k_frames=20):
        """
        Generate and visualize explanations for an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            target_class (int, optional): Target class for explanation
            save_path (str, optional): Path to save visualization
            top_k_frames (int): Number of top frames to highlight
            
        Returns:
            Dict: Explanation results
        """
        print(f"Visualizing explanations for {os.path.basename(audio_path)}")
        
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=16000)
        
        # Generate explanations
        explanations = self.explain(
            torch.tensor(waveform, dtype=torch.float32),
            target_class=target_class,
            mode='frame',
            aggregate_frames=None
        )
        
        # Extract frame attributions and predicted class
        attributions = explanations['attributions']
        pred_class = explanations['prediction']['class']
        pred_prob = explanations['prediction']['probability']
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Audio waveform
        plt.subplot(3, 1, 1)
        times = np.arange(len(waveform)) / sr
        plt.plot(times, waveform, color='blue', alpha=0.7)
        plt.title(f"Audio Waveform - Predicted: Class {pred_class} ({pred_prob:.2f})")
        plt.ylabel("Amplitude")
        plt.grid(alpha=0.3)
        
        # Plot 2: Spectrogram
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram")
        
        # Plot 3: Attributions
        plt.subplot(3, 1, 3)
        
        # Aggregate attributions for visualization
        if attributions.ndim > 1:
            # Take absolute value and mean across batch dimension
            attr_agg = np.abs(attributions).mean(axis=0)
        else:
            attr_agg = np.abs(attributions)
        
        # Normalize and plot
        attr_norm = attr_agg / (attr_agg.max() + 1e-10)
        plt.plot(times, attr_norm, color='red', alpha=0.8)
        
        # Highlight top k frames
        if top_k_frames > 0 and top_k_frames < len(attr_norm):
            top_indices = np.argsort(attr_norm)[-top_k_frames:]
            plt.scatter(times[top_indices], attr_norm[top_indices], 
                       color='darkred', s=30, zorder=10)
        
        plt.title("Frame-wise GradientSHAP Attribution")
        plt.ylabel("Normalized Attribution")
        plt.xlabel("Time (s)")
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return explanations


class WavLMGradientSHAPDemo:
    """
    Demonstration class for running GradientSHAP on WavLM with librosa datasets.
    
    This class provides a complete example workflow for:
    1. Loading a speech dataset
    2. Creating a classifier head
    3. Setting up the GradientSHAP explainer
    4. Generating and visualizing explanations
    """
    
    def __init__(
        self,
        model_name='microsoft/wavlm-base-plus',
        data_path=None,
        num_classes=8,
        device=None,
        batch_size=4,
        max_samples=100
    ):
        """
        Initialize the demo.
        
        Args:
            model_name (str): HuggingFace model name for WavLM
            data_path (str, optional): Path to audio dataset
            num_classes (int): Number of target classes (e.g., emotions)
            device (str): Device to use ('cuda' or 'cpu')
            batch_size (int): Batch size for processing
            max_samples (int): Maximum number of samples to use
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load WavLM model and feature extractor
        print(f"Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Create simple classifier head
        hidden_size = self.model.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_classes)
        ).to(self.device)
        
        # Create GradientSHAP explainer
        self.explainer = GradientSHAP(
            model_name=model_name,
            device=self.device,
            num_samples=50,
            classifier=self.classifier,
            batch_size=batch_size
        )
        
        # Dataset and parameters
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.dataset = None
        self.loader = None
        
        # Load dataset if path provided
        if data_path:
            self.load_dataset()
            
    def load_dataset(self, data_path=None, labels=None):
        """
        Load the audio dataset.
        
        Args:
            data_path (str, optional): Path to dataset (uses self.data_path if None)
            labels (dict, optional): Dictionary mapping filenames to labels
            
        Returns:
            DataLoader: PyTorch DataLoader for the dataset
        """
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("Data path not specified")
            
        # Create dataset
        self.dataset = LibrosaAudioDataset(
            data_path=self.data_path,
            feature_extractor=self.feature_extractor,
            labels=labels,
            max_samples=self.max_samples
        )
        
        # Create dataloader
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        
        print(f"Dataset loaded with {len(self.dataset)} samples")
        return self.loader
    
    def run_demo(self, audio_file=None, target_class=None, save_dir=None):
        """
        Run the GradientSHAP demonstration.
        
        Args:
            audio_file (str, optional): Path to specific audio file to explain
            target_class (int, optional): Target class for explanation
            save_dir (str, optional): Directory to save results
            
        Returns:
            Dict: Explanation results
        """
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # If specific file provided, explain it
        if audio_file:
            print(f"Generating explanation for {os.path.basename(audio_file)}")
            
            # Set up save path if directory provided
            save_path = None
            if save_dir:
                basename = os.path.basename(audio_file)
                save_path = os.path.join(save_dir, f"explanation_{basename}.png")
            
            # Generate and visualize explanation
            explanation = self.explainer.visualize_explanations(
                audio_path=audio_file,
                target_class=target_class,
                save_path=save_path
            )
            
            return explanation
        
        # Otherwise use dataset
        elif self.dataset and self.loader:
            results = []
            
            # Process a few samples
            for i, batch in enumerate(self.loader):
                if i >= 3:  # Limit to 3 samples for demonstration
                    break
                
                input_values = batch['input_values'][0]  # First sample in batch
                label = batch['label'][0].item() if batch['label'][0].item() >= 0 else None
                filename = batch['filename'][0]
                
                print(f"Generating explanation for {filename}")
                
                # Set up save path if directory provided
                save_path = None
                if save_dir:
                    save_path = os.path.join(save_dir, f"explanation_{i}_{filename}.png")
                
                # Convert to numpy
                input_np = input_values.cpu().numpy()
                
                # Create temporary audio file
                tmp_path = os.path.join(save_dir or '.', f"tmp_{filename}")
                librosa.output.write_wav(tmp_path, input_np, sr=16000, norm=False)
                
                try:
                    # Generate and visualize explanation
                    explanation = self.explainer.visualize_explanations(
                        audio_path=tmp_path,
                        target_class=label if label is not None else target_class,
                        save_path=save_path
                    )
                    
                    results.append({
                        'filename': filename,
                        'explanation': explanation
                    })
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            return results
        
        else:
            print("No audio file or dataset provided")
            return None


# Example usage
if __name__ == "__main__":
    # Example 1: Simple usage with a single audio file
    explainer = GradientSHAP(
        model_name='microsoft/wavlm-base-plus',
        num_samples=50,
        batch_size=4
    )
    
    # Replace with path to your audio file
    audio_file = "example_audio.wav"
    if os.path.exists(audio_file):
        explainer.visualize_explanations(
            audio_path=audio_file,
            target_class=0,  # Emotion class (0=neutral)
            save_path="gradientshap_explanation.png"
        )
    
    # Example 2: Demo with dataset
    demo = WavLMGradientSHAPDemo(
        model_name='microsoft/wavlm-base-plus',
        data_path="./audio_dataset",  # Replace with your dataset path
        num_classes=8  # For emotion recognition (8 emotions)
    )
    
    # Run demo if dataset exists
    if os.path.exists("./audio_dataset"):
        demo.run_demo(save_dir="./results")
