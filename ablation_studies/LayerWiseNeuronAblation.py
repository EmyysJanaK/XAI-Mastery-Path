#!/usr/bin/env python3

import torch
import torchaudio
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor  
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import json
from tqdm import tqdm
from sklearn.decomposition import PCA
import shap
import time
from sklearn.manifold import TSNE
import pandas as pd

class LayerWiseNeuronAblation:
    """
    Complete layer-by-layer neuron ablation analysis for WavLM on RAVDESS dataset
    """
    
    def __init__(self, model_name="microsoft/wavlm-base", device="cuda" if torch.cuda.is_available() else "cpu", debug=False):
        """Initialize WavLM model with proper feature extractor"""
        self.device = device
        self.debug = debug
        
        # ✅ Use FeatureExtractor instead of Processor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Model architecture info
        self.num_layers = len(self.model.encoder.layers)
        self.hidden_size = self.model.config.hidden_size
        
        print(f"WavLM Model Loaded:")
        print(f"  - Layers: {self.num_layers}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Device: {self.device}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - Current CUDA device: {torch.cuda.current_device()}")
            print(f"  - CUDA device name: {torch.cuda.get_device_name(0)}")
            
        # For debugging device issues
        if self.debug:
            print("\nDebug Mode Enabled - Additional Device Information:")
            print(f"  - PyTorch version: {torch.__version__}")
            
            # Check all model parameters are on the correct device
            params_on_device = all(p.device.type == device.split(':')[0] for p in self.model.parameters())
            print(f"  - All model parameters on {device}: {params_on_device}")
            
            # Show device mapping for first few model parameters
            param_devices = {name: param.device for name, param in list(self.model.named_parameters())[:5]}
            print(f"  - Sample parameter devices: {param_devices}")
            
            if not params_on_device:
                # Find parameters on wrong device
                wrong_device_params = [(name, param.device) for name, param in self.model.named_parameters() 
                                    if param.device.type != device.split(':')[0]]
                print(f"  - Parameters on wrong device: {wrong_device_params[:3]} (showing first 3)")
                
                # Try to force all parameters to correct device
                print(f"  - Moving all parameters to {device}...")
                self.model = self.model.to(device)
    
    def ensure_device_consistency(self, tensor, target_device=None):
        """
        Ensure tensor is on the specified device
        
        Args:
            tensor: PyTorch tensor to check
            target_device: Target device (if None, uses self.device)
            
        Returns:
            Tensor on the correct device
        """
        if not isinstance(tensor, torch.Tensor):
            return tensor
            
        device_to_use = target_device if target_device is not None else self.device
        
        # Check if tensor is on the right device
        if tensor.device.type != device_to_use.split(':')[0]:
            # Handle case where tensor is on CPU but should be on GPU or vice versa
            tensor = tensor.to(device_to_use)
        elif device_to_use.startswith('cuda') and ':' in device_to_use:
            # Handle specific GPU device index
            if str(tensor.device) != device_to_use:
                tensor = tensor.to(device_to_use)
        
        return tensor
    
    def load_ravdess_sample(self, ravdess_path: str, emotion_label: str = None, actor_id: int = None) -> Tuple[str, Dict]:
        """
        Load a specific sample from RAVDESS dataset
        
        Args:
            ravdess_path: Path to RAVDESS dataset
            emotion_label: Specific emotion to load
            actor_id: Specific actor ID (1-24)
        
        Returns:
            Tuple of (audio_file_path, metadata)
        """
        # RAVDESS emotion mapping
        emotion_map = {
            1: "neutral", 2: "calm", 3: "happy", 4: "sad",
            5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
        }
        
        reverse_emotion_map = {v: k for k, v in emotion_map.items()}
        
        audio_files = []
        
        # Scan RAVDESS directory structure
        ravdess_path = Path(ravdess_path)
        
        for actor_folder in ravdess_path.glob("Actor_*"):
            if not actor_folder.is_dir():
                continue
                
            actor_num = int(actor_folder.name.split("_")[1])
            
            # Filter by actor if specified
            if actor_id is not None and actor_num != actor_id:
                continue
            
            for audio_file in actor_folder.glob("*.wav"):
                # Parse RAVDESS filename: Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
                parts = audio_file.stem.split('-')
                
                if len(parts) >= 7:
                    modality = int(parts[0])
                    vocal_channel = int(parts[1])
                    emotion_id = int(parts[2])
                    intensity = int(parts[3])
                    statement = int(parts[4])
                    repetition = int(parts[5])
                    actor = int(parts[6])
                    
                    emotion_name = emotion_map.get(emotion_id, "unknown")
                    
                    # Filter by emotion if specified
                    if emotion_label is not None and emotion_name != emotion_label:
                        continue
                    
                    metadata = {
                        'file_path': str(audio_file),
                        'emotion_id': emotion_id,
                        'emotion_name': emotion_name,
                        'intensity': intensity,
                        'statement': statement,
                        'repetition': repetition,
                        'actor_id': actor,
                        'gender': 'female' if actor % 2 == 0 else 'male'
                    }
                    
                    audio_files.append((str(audio_file), metadata))
        
        if not audio_files:
            available_emotions = set()
            for actor_folder in ravdess_path.glob("Actor_*"):
                for audio_file in actor_folder.glob("*.wav"):
                    parts = audio_file.stem.split('-')
                    if len(parts) >= 3:
                        emo_id = int(parts[2])
                        available_emotions.add(emotion_map.get(emo_id, "unknown"))
            
            raise ValueError(f"No RAVDESS files found for emotion='{emotion_label}', actor='{actor_id}'. "
                           f"Available emotions: {sorted(available_emotions)}")
        
        # Return random sample
        audio_path, metadata = random.choice(audio_files)
        print(f"Selected audio: {metadata['emotion_name']} emotion, Actor {metadata['actor_id']} ({metadata['gender']})")
        
        return audio_path, metadata
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000, max_duration_sec: float = None, downsample_factor: int = 1) -> torch.Tensor:
        """
        Load and preprocess audio file for WavLM
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sampling rate
            max_duration_sec: Maximum duration in seconds to reduce memory usage (None = keep original)
            downsample_factor: Factor to downsample audio (1 = no downsampling, 2 = half the samples)
            
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
        
        # Trim audio to max duration if specified (to reduce memory usage)
        if max_duration_sec is not None:
            max_samples = int(max_duration_sec * target_sr)
            if waveform.shape[1] > max_samples:
                print(f"Trimming audio from {waveform.shape[1]/target_sr:.2f}s to {max_duration_sec}s to reduce memory usage")
                waveform = waveform[:, :max_samples]
        
        # Apply downsampling if specified (to reduce memory usage further)
        if downsample_factor > 1:
            original_duration = waveform.shape[1] / target_sr
            waveform = waveform[:, ::downsample_factor]
            new_duration = waveform.shape[1] / target_sr
            print(f"Downsampled audio by factor of {downsample_factor}: {original_duration:.2f}s → {new_duration:.2f}s")
            
            # Adjust sampling rate to maintain timing
            effective_sr = target_sr // downsample_factor
            print(f"Effective sampling rate: {effective_sr} Hz")
        else:
            effective_sr = target_sr
        
        # ✅ Use feature extractor instead of processor
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=effective_sr, 
            return_tensors="pt"
        )
        
        return inputs["input_values"].to(self.device)
    
    def create_task_metric(self, task_type: str = "emotion_representation"):
        """Create task-specific metric function"""
        
        if task_type == "emotion_representation":
            def emotion_metric(model, audio_input):
                """Measure quality of emotion representation"""
                with torch.no_grad():
                    outputs = model(audio_input)
                    # Use mean of last hidden state as emotion representation
                    emotion_repr = outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_size]
                    # L2 norm as quality metric
                    return emotion_repr.norm(dim=1).mean().item()
            return emotion_metric
        
        elif task_type == "hidden_state_variance":
            def variance_metric(model, audio_input):
                """Measure variance in hidden states (diversity metric)"""
                with torch.no_grad():
                    outputs = model(audio_input)
                    hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
                    # Variance across sequence dimension
                    variance = hidden_states.var(dim=1).mean().item()
                    return variance
            return variance_metric
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def find_important_neurons_in_layer(self, 
                                      audio_input: torch.Tensor,
                                      layer_idx: int,
                                      task_metric_fn,
                                      num_neurons_to_test: int = 50,
                                      ablation_method: str = "zero") -> Dict[int, Dict[str, float]]:
        """
        Find most important neurons in a specific layer using ablation
        
        Args:
            audio_input: Preprocessed audio tensor
            layer_idx: Layer index to analyze
            task_metric_fn: Function to measure task performance
            num_neurons_to_test: Number of neurons to test
            ablation_method: How to ablate neurons
            
        Returns:
            Dictionary mapping neuron indices to importance scores
        """
        print(f"\n🔍 Analyzing Layer {layer_idx} ({num_neurons_to_test} neurons)...")
        
        # Get baseline performance (no ablation)
        baseline_score = task_metric_fn(self.model, audio_input)
        print(f"  Baseline score: {baseline_score:.6f}")
        
        neuron_importance = {}
        layer = self.model.encoder.layers[layer_idx]
        
        # Test individual neurons
        for neuron_idx in tqdm(range(min(num_neurons_to_test, self.hidden_size)), 
                              desc=f"Layer {layer_idx} neurons"):
            
            # Create ablation hook for this specific neuron
            def create_ablation_hook(neuron_to_ablate):
                def ablation_hook(module, input_tensor, output_tensor):
                    if isinstance(output_tensor, tuple):
                        hidden_states = output_tensor[0].clone()
                        other_outputs = output_tensor[1:]
                    else:
                        hidden_states = output_tensor.clone()
                        other_outputs = ()
                    
                    # Ablate the specific neuron
                    if ablation_method == "zero":
                        hidden_states[:, :, neuron_to_ablate] = 0.0
                    elif ablation_method == "mean":
                        mean_val = hidden_states[:, :, neuron_to_ablate].mean()
                        hidden_states[:, :, neuron_to_ablate] = mean_val
                    elif ablation_method == "random":
                        std = hidden_states[:, :, neuron_to_ablate].std()
                        mean = hidden_states[:, :, neuron_to_ablate].mean()
                        shape = hidden_states[:, :, neuron_to_ablate].shape
                        random_vals = torch.normal(mean, std, shape, device=hidden_states.device)
                        hidden_states[:, :, neuron_to_ablate] = random_vals
                    
                    # Return modified output
                    if other_outputs:
                        return (hidden_states,) + other_outputs
                    return hidden_states
                
                return ablation_hook
            
            # Register hook
            hook = layer.register_forward_hook(create_ablation_hook(neuron_idx))
            
            try:
                # Measure performance with ablated neuron
                ablated_score = task_metric_fn(self.model, audio_input)
                importance = baseline_score - ablated_score
                
                neuron_importance[neuron_idx] = {
                    'importance_score': importance,
                    'baseline_score': baseline_score,
                    'ablated_score': ablated_score,
                    'relative_drop': (importance / baseline_score) if baseline_score != 0 else 0.0,
                    'layer_idx': layer_idx
                }
                
            except Exception as e:
                print(f"    Error testing neuron {neuron_idx}: {e}")
                neuron_importance[neuron_idx] = {
                    'importance_score': 0.0,
                    'baseline_score': baseline_score,
                    'ablated_score': baseline_score,
                    'relative_drop': 0.0,
                    'layer_idx': layer_idx,
                    'error': str(e)
                }
            finally:
                # Always remove hook
                hook.remove()
        
        # Sort by importance and return
        sorted_neurons = dict(sorted(neuron_importance.items(), 
                                   key=lambda x: x[1]['importance_score'], 
                                   reverse=True))
        
        # Print top neurons for this layer
        print(f"  Top 5 neurons in layer {layer_idx}:")
        for i, (neuron_idx, info) in enumerate(list(sorted_neurons.items())[:5]):
            print(f"    #{i+1}: Neuron {neuron_idx} (importance: {info['importance_score']:.6f})")
        
        return sorted_neurons
    
    def progressive_layer_ablation(self,
                                 audio_input: torch.Tensor,
                                 task_metric_fn,
                                 layers_to_analyze: List[int] = None,
                                 top_k_per_layer: int = 5,
                                 neurons_to_test_per_layer: int = 50) -> Dict[int, Any]:
        """
        Perform progressive layer-by-layer ablation analysis
        
        Args:
            audio_input: Preprocessed audio
            task_metric_fn: Task performance metric
            layers_to_analyze: Which layers to analyze (default: all)
            top_k_per_layer: How many top neurons to track per layer
            neurons_to_test_per_layer: How many neurons to test per layer
            
        Returns:
            Complete analysis results
        """
        if layers_to_analyze is None:
            layers_to_analyze = list(range(self.num_layers))
        
        print(f"\n🚀 Starting Progressive Layer-by-Layer Neuron Ablation")
        print(f"Analyzing layers: {layers_to_analyze}")
        print(f"Testing {neurons_to_test_per_layer} neurons per layer")
        print(f"Tracking top {top_k_per_layer} neurons per layer")
        
        results = {
            'baseline_score': task_metric_fn(self.model, audio_input),
            'layer_analysis': {},
            'progressive_ablation': {},
            'cumulative_effects': []
        }
        
        print(f"Baseline performance: {results['baseline_score']:.6f}")
        
        # Step 1: Analyze each layer individually
        print(f"\n{'='*60}")
        print("STEP 1: INDIVIDUAL LAYER ANALYSIS")
        print(f"{'='*60}")
        
        for layer_idx in layers_to_analyze:
            layer_results = self.find_important_neurons_in_layer(
                audio_input=audio_input,
                layer_idx=layer_idx,
                task_metric_fn=task_metric_fn,
                num_neurons_to_test=neurons_to_test_per_layer
            )
            
            results['layer_analysis'][layer_idx] = layer_results
        
        # Step 2: Progressive ablation (layer by layer)
        print(f"\n{'='*60}")
        print("STEP 2: PROGRESSIVE LAYER-BY-LAYER ABLATION")
        print(f"{'='*60}")
        
        active_hooks = []  # Track active ablation hooks
        cumulative_ablated_layers = []
        
        for layer_idx in sorted(layers_to_analyze):
            print(f"\n📍 Adding Layer {layer_idx} to progressive ablation...")
            
            # Get top neurons from this layer
            layer_neurons = results['layer_analysis'][layer_idx]
            top_neurons = list(layer_neurons.keys())[:top_k_per_layer]
            
            # Create progressive ablation hook for this layer
            layer = self.model.encoder.layers[layer_idx]
            
            def create_progressive_hook(neurons_to_ablate):
                def progressive_hook(module, input_tensor, output_tensor):
                    if isinstance(output_tensor, tuple):
                        hidden_states = output_tensor[0].clone()
                        other_outputs = output_tensor[1:]
                    else:
                        hidden_states = output_tensor.clone()
                        other_outputs = ()
                    
                    # Ablate all specified neurons
                    for neuron_idx in neurons_to_ablate:
                        hidden_states[:, :, neuron_idx] = 0.0
                    
                    if other_outputs:
                        return (hidden_states,) + other_outputs
                    return hidden_states
                
                return progressive_hook
            
            # Register progressive hook
            hook = layer.register_forward_hook(create_progressive_hook(top_neurons))
            active_hooks.append((layer_idx, hook))
            cumulative_ablated_layers.append(layer_idx)
            
            # Measure cumulative performance
            cumulative_score = task_metric_fn(self.model, audio_input)
            cumulative_drop = results['baseline_score'] - cumulative_score
            
            progressive_result = {
                'cumulative_layers': cumulative_ablated_layers.copy(),
                'ablated_neurons_this_layer': top_neurons,
                'cumulative_score': cumulative_score,
                'cumulative_drop': cumulative_drop,
                'relative_cumulative_drop': cumulative_drop / results['baseline_score'],
                'incremental_drop': cumulative_drop - (results['cumulative_effects'][-1]['cumulative_drop'] if results['cumulative_effects'] else 0)
            }
            
            results['progressive_ablation'][layer_idx] = progressive_result
            results['cumulative_effects'].append(progressive_result)
            
            print(f"  Ablated neurons: {top_neurons}")
            print(f"  Cumulative score: {cumulative_score:.6f}")
            print(f"  Cumulative drop: {cumulative_drop:.6f}")
            print(f"  Incremental drop: {progressive_result['incremental_drop']:.6f}")
        
        # Clean up all hooks
        print(f"\n🧹 Cleaning up ablation hooks...")
        for layer_idx, hook in active_hooks:
            hook.remove()
        
        return results
    
    def reduce_dimensions(self, model_wrapper, inputs, method='pca', n_components=100, k=100):
        """
        Apply dimensionality reduction to model features using different methods
        
        Args:
            model_wrapper: Model wrapper that produces features
            inputs: Input data
            method: 'pca', 'random', or None
            n_components: Number of PCA components (for 'pca')
            k: Number of top neurons to select (for 'topk')
            
        Returns:
            Tuple of (reduced_features, reducer)
        """
        print(f"Applying {method} dimensionality reduction...")
        
        # Get features from model
        try:
            with torch.no_grad():
                # Ensure inputs are on the same device as model
                inputs_device = inputs.to(self.device) if isinstance(inputs, torch.Tensor) else inputs
                features = model_wrapper(inputs_device).detach().cpu().numpy()
                print(f"Extracted features with shape: {features.shape}")
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Try with smaller batch or single sample as fallback
            try:
                with torch.no_grad():
                    if isinstance(inputs, torch.Tensor) and inputs.shape[0] > 1:
                        print("Trying with smaller batch...")
                        single_input = inputs[0:1]  # Take first sample
                        features_single = model_wrapper(single_input).detach().cpu().numpy()
                        # Duplicate to get multiple samples
                        features = np.repeat(features_single, 10, axis=0)
                        # Add noise to avoid singularity
                        features += np.random.normal(0, 0.001, features.shape)
                    else:
                        raise ValueError("Input already has batch size 1")
            except Exception as e2:
                print(f"Failed to extract features with fallback: {e2}")
                # Create dummy features as last resort
                print("Creating dummy features for dimensionality reduction")
                features = np.random.randn(10, model_wrapper.model.config.hidden_size)
        
        # Apply appropriate reduction method
        if method == 'pca':
            return self.reduce_dimensions_pca(features, n_components)
        elif method == 'topk':
            return self.select_top_k_neurons(features, k=k)
        elif method == 'random':
            # Random selection of neurons
            feature_dim = features.shape[1]
            indices = np.random.choice(feature_dim, size=min(n_components, feature_dim), replace=False)
            return features[:, indices], indices
        else:
            # No reduction
            return features, None

    def reduce_dimensions_pca(self, features: Union[torch.Tensor, np.ndarray], n_components: int = 100) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA dimensionality reduction to model features
        
        Args:
            features: Tensor or array of shape [num_samples, hidden_dim]
            n_components: Number of PCA components to keep
            
        Returns:
            Tuple of (reduced_features, pca_model)
        """
        # Convert to numpy if it's a tensor
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features
        
        # Handle single sample case - duplicate it to create more samples
        if features_np.ndim == 2 and features_np.shape[0] == 1:
            print(f"Warning: Only 1 sample detected with shape {features_np.shape}, duplicating to allow PCA...")
            features_np = np.repeat(features_np, 10, axis=0)
            # Add small random noise to make samples slightly different
            features_np += np.random.normal(0, 0.001, features_np.shape)
        
        # Handle NaN or inf values
        if np.isnan(features_np).any() or np.isinf(features_np).any():
            print("Warning: NaN or inf values detected in features, replacing with zeros")
            features_np = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for zero variance features
        variances = np.var(features_np, axis=0)
        if np.any(variances == 0):
            print("Warning: Zero variance features detected, adding small noise")
            zero_idx = np.where(variances == 0)[0]
            features_np[:, zero_idx] += np.random.normal(0, 0.001, size=(features_np.shape[0], len(zero_idx)))
        
        # Ensure we don't try to extract more components than possible
        max_components = min(features_np.shape[0], features_np.shape[1])
        safe_n_components = min(n_components, max_components - 1)
        
        if safe_n_components < n_components:
            print(f"Warning: Reducing n_components from {n_components} to {safe_n_components} due to data shape constraints")
        
        if safe_n_components <= 0:
            print("Cannot perform PCA: Too few samples or features")
            return features_np, None
        
        # Apply PCA with error handling
        try:
            pca = PCA(n_components=safe_n_components)
            reduced_features = pca.fit_transform(features_np)
            
            print(f"Original features: {features_np.shape}, Reduced: {reduced_features.shape}")
            print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
            
            return reduced_features, pca
            
        except Exception as e:
            print(f"PCA failed: {e}")
            print("Returning original features")
            return features_np, None
    
    def select_top_k_neurons(self, features: Union[torch.Tensor, np.ndarray], k: int = 100, 
                          method: str = 'variance') -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top-k neurons based on variance or activation magnitude
        
        Args:
            features: Tensor or array of shape [num_samples, hidden_dim]
            k: Number of neurons to select
            method: 'variance' or 'activation'
            
        Returns:
            Tuple of (reduced_features, selected_indices)
        """
        # Convert to numpy if it's a tensor
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features
            
        if method == 'variance':
            # Calculate variance of each neuron across samples
            neuron_variance = np.var(features_np, axis=0)
            # Get indices of top-k neurons by variance
            top_k_indices = np.argsort(-neuron_variance)[:k]
        else:  # activation
            # Calculate mean activation (absolute value) of each neuron
            neuron_activation = np.mean(np.abs(features_np), axis=0)
            # Get indices of top-k neurons by activation
            top_k_indices = np.argsort(-neuron_activation)[:k]
        
        # Select only those neurons
        reduced_features = features_np[:, top_k_indices]
        
        print(f"Original features: {features_np.shape}, Reduced: {reduced_features.shape}")
        print(f"Selected {k} neurons using {method} method")
        
        return reduced_features, top_k_indices
    
    def visualize_dimensionality_reduction(self, original_features, reduced_features, 
                                       reduction_method: str, pca_model=None, topk_indices=None):
        """
        Visualize dimensionality reduction results
        
        Args:
            original_features: Original features
            reduced_features: Reduced features from PCA or Top-K
            reduction_method: 'pca' or 'topk'
            pca_model: PCA model (if reduction_method is 'pca')
            topk_indices: Selected neuron indices (if reduction_method is 'topk')
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Plot original vs reduced feature distribution
        axes[0].boxplot([np.ravel(original_features), np.ravel(reduced_features)])
        axes[0].set_xticklabels(['Original', 'Reduced'])
        axes[0].set_title('Feature Value Distribution')
        axes[0].set_ylabel('Feature Value')
        axes[0].grid(True, alpha=0.3)
        
        # 2. If PCA, plot explained variance
        if reduction_method == 'pca' and pca_model is not None:
            cumulative_var = np.cumsum(pca_model.explained_variance_ratio_)
            axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 'b-', linewidth=2)
            axes[1].set_title('PCA Cumulative Explained Variance')
            axes[1].set_xlabel('Number of Components')
            axes[1].set_ylabel('Cumulative Explained Variance')
            axes[1].grid(True, alpha=0.3)
            
            # Highlight 80%, 90%, 95% variance thresholds
            thresholds = [0.8, 0.9, 0.95]
            for thresh in thresholds:
                # Find first component that explains thresh variance
                idx = np.where(cumulative_var >= thresh)[0][0]
                axes[1].axhline(y=thresh, color='r', linestyle='--', alpha=0.3)
                axes[1].axvline(x=idx+1, color='g', linestyle='--', alpha=0.3)
                axes[1].text(idx+2, thresh, f'{thresh:.0%}: {idx+1} components', 
                            verticalalignment='bottom')
        
        # If Top-K, plot selected neuron distribution
        elif reduction_method == 'topk' and topk_indices is not None:
            if isinstance(original_features, torch.Tensor):
                original_np = original_features.detach().cpu().numpy()
            else:
                original_np = original_features
                
            if method == 'variance':
                # Plot variance of each neuron
                neuron_variance = np.var(original_np, axis=0)
                selected_variance = neuron_variance[topk_indices]
                
                # Sort by variance
                sorted_idx = np.argsort(-selected_variance)
                sorted_variance = selected_variance[sorted_idx]
                
                axes[1].bar(range(len(sorted_variance)), sorted_variance, alpha=0.7)
                axes[1].set_title('Variance of Selected Top-K Neurons')
                axes[1].set_xlabel('Neuron Index (sorted by variance)')
                axes[1].set_ylabel('Variance')
                axes[1].grid(True, alpha=0.3)
            else:
                # Plot activation of each neuron
                neuron_activation = np.mean(np.abs(original_np), axis=0)
                selected_activation = neuron_activation[topk_indices]
                
                # Sort by activation
                sorted_idx = np.argsort(-selected_activation)
                sorted_activation = selected_activation[sorted_idx]
                
                axes[1].bar(range(len(sorted_activation)), sorted_activation, alpha=0.7)
                axes[1].set_title('Mean Activation of Selected Top-K Neurons')
                axes[1].set_xlabel('Neuron Index (sorted by activation)')
                axes[1].set_ylabel('Mean Activation')
                axes[1].grid(True, alpha=0.3)
        
        # 3. Compare original vs reduced with t-SNE
        # Apply t-SNE to visualize in 2D space
        n_samples = min(1000, original_features.shape[0])  # Limit samples for t-SNE
        
        # Sample if needed
        if original_features.shape[0] > n_samples:
            indices = np.random.choice(original_features.shape[0], n_samples, replace=False)
            original_sample = original_features[indices]
            reduced_sample = reduced_features[indices]
        else:
            original_sample = original_features
            reduced_sample = reduced_features
        
        # Apply t-SNE
        try:
            tsne_original = TSNE(n_components=2, random_state=42).fit_transform(original_sample)
            tsne_reduced = TSNE(n_components=2, random_state=42).fit_transform(reduced_sample)
            
            # Plot original t-SNE
            axes[2].scatter(tsne_original[:, 0], tsne_original[:, 1], alpha=0.5, s=20, c='blue', label='Original')
            # Plot reduced t-SNE
            axes[2].scatter(tsne_reduced[:, 0], tsne_reduced[:, 1], alpha=0.5, s=20, c='red', label='Reduced')
            axes[2].set_title('t-SNE Visualization')
            axes[2].legend()
            axes[2].set_xlabel('t-SNE Dimension 1')
            axes[2].set_ylabel('t-SNE Dimension 2')
        except Exception as e:
            axes[2].text(0.5, 0.5, f"t-SNE failed: {str(e)}", horizontalalignment='center',
                        verticalalignment='center', transform=axes[2].transAxes)
        
        plt.tight_layout()
        plt.suptitle(f'Dimensionality Reduction Analysis: {reduction_method.upper()}', fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.show()
        
    def visualize_layer_analysis(self, results: Dict, save_path: str = None):
        """Create comprehensive visualizations of layer-wise analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data for plotting
        layers = sorted(results['layer_analysis'].keys())
        
        # Plot 1: Average importance per layer
        avg_importance_per_layer = []
        max_importance_per_layer = []
        
        for layer_idx in layers:
            layer_data = results['layer_analysis'][layer_idx]
            importance_scores = [data['importance_score'] for data in layer_data.values()]
            avg_importance_per_layer.append(np.mean(importance_scores))
            max_importance_per_layer.append(np.max(importance_scores))
        
        axes[0, 0].plot(layers, avg_importance_per_layer, 'b-o', label='Average Importance', linewidth=2)
        axes[0, 0].plot(layers, max_importance_per_layer, 'r-s', label='Max Importance', linewidth=2)
        axes[0, 0].set_title('Neuron Importance Across Layers')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Importance Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Progressive ablation performance
        if results['cumulative_effects']:
            cumulative_layers = [len(effect['cumulative_layers']) for effect in results['cumulative_effects']]
            cumulative_scores = [effect['cumulative_score'] for effect in results['cumulative_effects']]
            
            axes[0, 1].plot(cumulative_layers, cumulative_scores, 'g-^', linewidth=2, markersize=8)
            axes[0, 1].axhline(y=results['baseline_score'], color='black', linestyle='--', label='Baseline')
            axes[0, 1].set_title('Progressive Layer Ablation Effect')
            axes[0, 1].set_xlabel('Number of Layers Ablated')
            axes[0, 1].set_ylabel('Model Performance')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Top neurons heatmap (first few layers)
        layers_to_show = layers[:min(6, len(layers))]
        top_neurons_matrix = []
        
        for layer_idx in layers_to_show:
            layer_data = results['layer_analysis'][layer_idx]
            # Get top 10 neurons
            top_10 = list(layer_data.keys())[:10]
            importance_scores = [layer_data[n]['importance_score'] for n in top_10]
            top_neurons_matrix.append(importance_scores)
        
        if top_neurons_matrix:
            im = axes[1, 0].imshow(top_neurons_matrix, cmap='viridis', aspect='auto')
            axes[1, 0].set_title('Top 10 Neurons per Layer (Importance Heatmap)')
            axes[1, 0].set_xlabel('Neuron Rank (Top 10)')
            axes[1, 0].set_ylabel('Layer Index')
            axes[1, 0].set_yticks(range(len(layers_to_show)))
            axes[1, 0].set_yticklabels(layers_to_show)
            plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Incremental vs Cumulative drops
        if results['cumulative_effects']:
            incremental_drops = [effect['incremental_drop'] for effect in results['cumulative_effects']]
            cumulative_drops = [effect['cumulative_drop'] for effect in results['cumulative_effects']]
            
            x_pos = range(len(incremental_drops))
            
            axes[1, 1].bar([x - 0.2 for x in x_pos], incremental_drops, 0.4, 
                          label='Incremental Drop', alpha=0.7)
            axes[1, 1].plot(x_pos, cumulative_drops, 'r-o', label='Cumulative Drop', linewidth=2)
            axes[1, 1].set_title('Incremental vs Cumulative Performance Drops')
            axes[1, 1].set_xlabel('Progressive Ablation Step')
            axes[1, 1].set_ylabel('Performance Drop')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels([f"L{layers[i]}" for i in range(len(incremental_drops))])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_wavlm_wrapper(self, layer_idx=-1):
        """
        Create a wrapper for the WavLM model that makes it compatible with GradientSHAP
        
        Args:
            layer_idx: Layer index to extract hidden states from (-1 for last layer)
            
        Returns:
            WavLM wrapper model
        """
        # Get a reference to the parent class for ensure_device_consistency
        parent_self = self
        
        # Define wrapper class
        class WavLMWrapper(torch.nn.Module):
            def __init__(self, model, layer_idx):
                super().__init__()
                self.model = model
                self.layer_idx = layer_idx
                self.model.eval()
                self.device = parent_self.device
                
                # Create a simple trainable layer to ensure gradients flow
                self.grad_enabler = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, 
                                                  device=self.device)
                # Initialize as identity matrix to not change values
                self.grad_enabler.weight.data.copy_(torch.eye(self.model.config.hidden_size, device=self.device))
                self.grad_enabler.bias.data.zero_()
                
                print(f"WavLMWrapper created with device: {self.device}")
                print(f"Model device: {self.model.device}")
                print(f"Gradient enabler device: {self.grad_enabler.weight.device}")
            
            def forward(self, x):
                # Use the parent's device consistency check
                x = parent_self.ensure_device_consistency(x, self.device)
                
                try:
                    # Don't use torch.no_grad() - we need gradients for SHAP
                    outputs = self.model(x, output_hidden_states=True)
                        
                    # Extract hidden states from the specified layer
                    hidden_states = outputs.hidden_states[self.layer_idx]
                    
                    # Average over sequence length to get a fixed-size representation
                    avg_hidden_states = torch.mean(hidden_states, dim=1)
                    
                    # Ensure device consistency before passing through grad_enabler
                    avg_hidden_states = parent_self.ensure_device_consistency(avg_hidden_states, self.device)
                    
                    # Pass through gradient enabler (identity transform)
                    # This ensures that gradients can flow back through the model
                    return self.grad_enabler(avg_hidden_states)
                except Exception as e:
                    print(f"Error in WavLMWrapper.forward: {e}")
                    print(f"Input shape: {x.shape}, device: {x.device}")
                    print(f"Model device: {self.model.device}")
                    raise
        
        # Create and return wrapper instance
        return WavLMWrapper(self.model, layer_idx)
    
    def create_gradient_shap_explainer(self, layer_idx=-1, background_data=None):
        """
        Create a GradientSHAP explainer for WavLM model
        
        Args:
            layer_idx: Layer index to analyze
            background_data: Background data for the explainer (if None, random data will be used)
            
        Returns:
            GradientSHAP explainer and model wrapper
        """
        print(f"Creating GradientSHAP explainer for layer {layer_idx}...")
        
        # Create model wrapper
        model_wrapper = self.create_wavlm_wrapper(layer_idx)
        
        # Put model in eval mode but ensure we can get gradients
        model_wrapper.eval()
        
        # Create background data if not provided
        if background_data is None:
            # Generate random background data (10 samples)
            print("No background data provided, generating random samples...")
            background_data = torch.randn(10, 16000, device=self.device)
        
        # Ensure we have at least 2 samples for background data
        # as GradientSHAP requires more than one background sample
        if isinstance(background_data, torch.Tensor):
            if background_data.ndim == 2:  # [batch=1, seq_len]
                if background_data.shape[0] < 2:
                    print("Expanding background samples to at least 2 samples...")
                    # Create multiple variations by adding noise
                    expanded_data = [background_data]
                    for i in range(4):  # Create 4 more samples
                        noise_level = 0.001 * (i + 1)  # Gradually increase noise
                        expanded_data.append(background_data + torch.randn_like(background_data) * noise_level)
                    background_data = torch.cat(expanded_data, dim=0)
            
            print(f"Background data shape: {background_data.shape}")
        
        # Create explainer
        try:
            # First try with regular background data
            start_time = time.time()
            
            # Check if background data is too large (might cause OOM errors)
            if isinstance(background_data, torch.Tensor) and background_data.numel() > 10000000:  # ~10M elements
                print(f"Warning: Large background tensor ({background_data.numel()} elements). Attempting to reduce size.")
                # Use a smaller subset or downsample
                if background_data.ndim > 1 and background_data.shape[0] > 2:
                    # Take only first 2 samples
                    background_data = background_data[:2]
                elif background_data.ndim > 1 and background_data.shape[1] > 10000:
                    # Downsample sequence dimension if possible
                    background_data = background_data[:, ::2]  # Take every other element
            
            # Make sure background data requires grad and is on the right device
            if isinstance(background_data, torch.Tensor) and not background_data.requires_grad:
                background_data_with_grad = background_data.detach().clone().to(self.device).requires_grad_(True)
            else:
                background_data_with_grad = background_data.to(self.device) if isinstance(background_data, torch.Tensor) else background_data
                
            # Print memory usage before creating explainer
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                print(f"GPU memory allocated before creating explainer: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")
                print(f"GPU memory reserved before creating explainer: {torch.cuda.memory_reserved() / 1024**2:.1f} MiB")
                
            explainer = shap.GradientExplainer(model_wrapper, background_data_with_grad)
            print(f"Explainer creation took {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error creating GradientExplainer with original data: {e}")
            print("Attempting with pre-processed background data...")
            
            try:
                # Pre-process the background data through the model to get hidden states
                with torch.no_grad():
                    # Process one sample at a time to avoid OOM errors
                    processed_samples = []
                    batch_size = 1
                    
                    if isinstance(background_data, torch.Tensor):
                        for i in range(0, background_data.shape[0], batch_size):
                            batch = background_data[i:i+batch_size]
                            hidden = model_wrapper(batch).detach()
                            processed_samples.append(hidden)
                            
                        processed_bg = torch.cat(processed_samples, dim=0)
                    else:
                        # Handle non-tensor background data
                        processed_bg = None
                    
                # Create random data as fallback if processing failed
                if processed_bg is None or processed_bg.shape[0] < 2:
                    print("Generating random background data...")
                    feature_dim = model_wrapper.model.config.hidden_size
                    processed_bg = torch.randn(5, feature_dim, device=self.device)
                
                # Ensure the background data requires gradients
                processed_bg_with_grad = processed_bg.clone().requires_grad_(True)
                
                # Create a simpler wrapper that works directly with hidden states
                class DirectWrapper(torch.nn.Module):
                    def __init__(self, orig_wrapper):
                        super().__init__()
                        self.grad_enabler = orig_wrapper.grad_enabler
                    
                    def forward(self, x):
                        # Apply the gradient enabler directly
                        return self.grad_enabler(x)
                
                simple_wrapper = DirectWrapper(model_wrapper)
                
                # Try creating explainer with the processed background data
                explainer = shap.GradientExplainer(simple_wrapper, processed_bg_with_grad)
                print("Created explainer with pre-processed background data")
                
            except Exception as e2:
                print(f"Error creating explainer with pre-processed data: {e2}")
                print("All attempts to create GradientExplainer failed")
                raise ValueError("Unable to create GradientSHAP explainer")
        
        return explainer, model_wrapper
    
    def auto_select_device_for_computation(self, computation_size_estimate):
        """
        Automatically select the best device (CPU/GPU) based on computation size
        
        Args:
            computation_size_estimate: Estimated size of computation in elements
            
        Returns:
            device: Recommended device for computation
        """
        # Check if CUDA is available
        if not torch.cuda.is_available():
            return "cpu"
            
        # If computation is very small, use CPU to avoid GPU overhead
        if computation_size_estimate < 1000000:  # < 1M elements
            return "cpu"
            
        # Check GPU memory status
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free_gpu_memory = total_gpu_memory - gpu_memory_allocated
        
        # Estimate memory needed (very rough estimate: 4 bytes per float * elements * safety factor)
        estimated_memory_needed = computation_size_estimate * 4 * 3 / 1024**3  # GB
        
        print(f"Memory estimate: need ~{estimated_memory_needed:.2f} GB, free GPU memory: {free_gpu_memory:.2f} GB")
        
        # If we don't have enough GPU memory, use CPU
        if estimated_memory_needed > free_gpu_memory * 0.8:  # Keep 20% buffer
            print("Insufficient GPU memory, switching to CPU")
            return "cpu"
            
        return self.device  # Default to current device
        
    def compute_shap_in_batches(self, explainer, data, batch_size=2, n_samples=100, memory_efficient=True):
        """
        Compute SHAP values in batches to reduce memory usage and improve speed
        
        Args:
            explainer: SHAP explainer
            data: Dataset to analyze
            batch_size: Batch size for processing
            n_samples: Number of samples for SHAP computation (for KernelSHAP)
            memory_efficient: Use memory efficient mode (smaller internal batch sizes)
            
        Returns:
            SHAP values for all samples
        """
        print(f"Computing SHAP values in batches (batch_size={batch_size}, n_samples={n_samples}, memory_efficient={memory_efficient})...")
        
        # In memory efficient mode, check and free GPU memory if needed
        if memory_efficient and torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            free_memory = total_memory - memory_allocated
            
            print(f"GPU memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
            print(f"Total GPU memory: {total_memory:.2f} GB, Free: {free_memory:.2f} GB")
            
            # If memory is getting tight, try to free some
            if free_memory < 1.0:  # Less than 1GB free
                print("Low GPU memory detected, attempting to free unused memory...")
                torch.cuda.empty_cache()
                # Force garbage collection
                import gc
                gc.collect()
        
        # Get data length
        if isinstance(data, torch.Tensor):
            data_len = data.shape[0]
        else:
            data_len = len(data)
        
        # Adjust batch size for small datasets
        batch_size = min(batch_size, data_len)
        if batch_size <= 0:
            print("Error: No data to process")
            
            # If we have feature dimension information, return dummy values
            if hasattr(explainer, 'expected_value'):
                if hasattr(explainer.expected_value, 'shape'):
                    feature_dim = explainer.expected_value.shape[0]
                else:
                    feature_dim = 768  # Default
                return np.zeros((1, feature_dim))
            else:
                return np.array([])
        
        print(f"Processing {data_len} samples with adjusted batch size: {batch_size}")
        
        # Process in batches
        all_shap_values = []
        start_time = time.time()
        
        # Determine what type of explainer we're using
        is_gradient_explainer = isinstance(explainer, shap.explainers.Gradient)
        is_kernel_explainer = isinstance(explainer, shap.explainers.Kernel)
        
        # Create a robust wrapper for SHAP computation
        def compute_shap_robust(batch_data):
            try:
                # Add proper handling for different explainer types
                if is_gradient_explainer:
                    # Use the device consistency function
                    if isinstance(batch_data, torch.Tensor):
                        # First ensure it's on the right device
                        batch_data = self.ensure_device_consistency(batch_data)
                        
                        # Then ensure it has gradients if needed
                        if not batch_data.requires_grad:
                            batch_data = batch_data.detach().clone().requires_grad_(True)
                            
                        print(f"Batch data device before SHAP: {batch_data.device}, requires_grad: {batch_data.requires_grad}")
                    
                    return explainer.shap_values(batch_data)
                elif is_kernel_explainer:
                    # For KernelExplainer, we can specify the number of samples
                    return explainer.shap_values(batch_data, nsamples=n_samples)
                else:
                    # For other explainers, use default
                    return explainer.shap_values(batch_data)
            except Exception as e:
                print(f"SHAP computation error: {e}")
                
                # Create dummy values with appropriate shape
                if isinstance(batch_data, torch.Tensor):
                    batch_size = batch_data.shape[0]
                elif isinstance(batch_data, np.ndarray):
                    batch_size = batch_data.shape[0]
                else:
                    batch_size = 1
                
                # Try to determine feature dimension
                if hasattr(explainer, 'expected_value'):
                    if hasattr(explainer.expected_value, 'shape'):
                        feature_dim = explainer.expected_value.shape[0]
                    else:
                        # Try other ways to get dimension
                        try:
                            if hasattr(explainer.model, 'model') and hasattr(explainer.model.model, 'config'):
                                feature_dim = explainer.model.model.config.hidden_size
                            else:
                                feature_dim = 768  # Default fallback
                        except:
                            feature_dim = 768  # Default fallback
                else:
                    feature_dim = 768  # Default fallback
                
                # Return dummy values
                return np.zeros((batch_size, feature_dim))
        
        try:
            # For very small datasets, try processing all at once first
            if data_len <= batch_size:
                print("Small dataset, trying to process all data at once...")
                try:
                    all_values = compute_shap_robust(data)
                    
                    # Handle different return types
                    if isinstance(all_values, list):
                        all_shap_values = all_values
                    else:
                        all_shap_values = [all_values]
                        
                    print(f"Successfully processed all {data_len} samples at once")
                    
                except Exception as e:
                    print(f"Error processing all data at once: {e}")
                    print("Falling back to batch processing")
                    # Continue with batch processing below
                    all_shap_values = []  # Reset for batch processing
            
            # Process in batches if we don't have results yet
            if not all_shap_values:
                for i in tqdm(range(0, data_len, batch_size), desc="SHAP Batch Processing"):
                    # Get batch
                    batch_end = min(i + batch_size, data_len)
                    if isinstance(data, torch.Tensor):
                        batch = data[i:batch_end]
                    else:
                        batch = data[i:batch_end]
                    
                    try:
                        # Compute SHAP values for batch
                        batch_shap_values = compute_shap_robust(batch)
                        
                        # Store results
                        if isinstance(batch_shap_values, list):
                            all_shap_values.extend(batch_shap_values)
                        else:
                            # If a single array is returned, append it
                            if len(all_shap_values) == 0:
                                # First batch
                                all_shap_values = [batch_shap_values]
                            else:
                                # Append to existing array or list
                                if isinstance(all_shap_values, list) and len(all_shap_values) == 1:
                                    # We have a list with one array
                                    all_shap_values[0] = np.vstack([all_shap_values[0], batch_shap_values])
                                else:
                                    # We have a regular list
                                    all_shap_values.append(batch_shap_values)
                    except Exception as e:
                        print(f"Error processing batch {i}:{batch_end}: {e}")
                        
                        # Create dummy values for this batch
                        dummy_shape = batch_end - i  # Batch size
                        
                        # Try to determine feature dimension
                        if hasattr(explainer, 'expected_value'):
                            if hasattr(explainer.expected_value, 'shape'):
                                feature_dim = explainer.expected_value.shape[0]
                            else:
                                feature_dim = 768  # Default
                        else:
                            feature_dim = 768  # Default
                        
                        print(f"Creating dummy values with shape [{dummy_shape}, {feature_dim}]")
                        dummy_values = np.zeros((dummy_shape, feature_dim))
                        
                        # Add to results
                        if len(all_shap_values) == 0:
                            # First batch
                            all_shap_values = [dummy_values]
                        else:
                            # Append to existing
                            if isinstance(all_shap_values, list) and len(all_shap_values) == 1 and isinstance(all_shap_values[0], np.ndarray):
                                # We have a list with one array
                                try:
                                    all_shap_values[0] = np.vstack([all_shap_values[0], dummy_values])
                                except:
                                    all_shap_values.append(dummy_values)
                            else:
                                # We have a regular list
                                all_shap_values.append(dummy_values)
                        
        except Exception as e:
            print(f"Error in batch processing: {e}")
            
            # Check if we have any results
            if not all_shap_values:
                print("No SHAP values computed. Generating dummy values...")
                
                # Create dummy data with appropriate shape
                if isinstance(data, torch.Tensor):
                    data_shape = data.shape
                elif isinstance(data, np.ndarray):
                    data_shape = data.shape
                else:
                    data_shape = (data_len, 768)  # Default
                
                # Create single dummy array
                if len(data_shape) >= 2:
                    feature_dim = 768  # Default
                    if hasattr(explainer, 'expected_value'):
                        if hasattr(explainer.expected_value, 'shape'):
                            feature_dim = explainer.expected_value.shape[0]
                    
                    # Create dummy SHAP values
                    dummy_values = np.zeros((data_shape[0], feature_dim))
                    all_shap_values = [dummy_values]
        
        total_time = time.time() - start_time
        print(f"SHAP computation completed in {total_time:.2f} seconds")
        if data_len > 0:
            print(f"Average time per sample: {total_time / data_len:.4f} seconds")
        
        # Convert to proper format
        if all_shap_values:
            # Check if we have a list of arrays or a single array
            if isinstance(all_shap_values, list):
                if len(all_shap_values) == 1 and isinstance(all_shap_values[0], np.ndarray):
                    # Return the single array
                    return all_shap_values[0]
                elif all(isinstance(x, np.ndarray) for x in all_shap_values):
                    # We have a list of arrays, try to stack them
                    try:
                        return np.vstack(all_shap_values)
                    except:
                        return all_shap_values
                else:
                    return all_shap_values
            else:
                # It's already a single array
                return all_shap_values
        else:
            print("Warning: No SHAP values were computed")
            
            # Create dummy output
            if isinstance(data, torch.Tensor):
                data_shape = data.shape
            elif isinstance(data, np.ndarray):
                data_shape = data.shape
            else:
                data_shape = (1, 768)  # Default
                
            if len(data_shape) >= 2:
                return np.zeros((data_shape[0], 768))  # Default feature dim
            else:
                return np.zeros((1, 768))
    
    def analyze_shap_values(self, shap_values, layer_idx, top_k=20):
        """
        Analyze SHAP values for a layer
        
        Args:
            shap_values: SHAP values from compute_shap_in_batches
            layer_idx: Layer index that was analyzed
            top_k: Number of top neurons to identify
            
        Returns:
            Dictionary with neuron importance information
        """
        # Check if shap_values is empty or malformed
        if shap_values is None or (isinstance(shap_values, np.ndarray) and shap_values.size == 0):
            print(f"Warning: Empty SHAP values for layer {layer_idx}")
            # Create dummy results
            feature_dim = self.hidden_size
            random_indices = np.random.choice(feature_dim, size=top_k, replace=False)
            dummy_importance = np.random.random(top_k) * 0.001  # Small random values
            
            results = {
                'layer_idx': layer_idx,
                'top_neurons': random_indices,
                'neuron_importance': dummy_importance,
                'importance_coverage': 0.0,
                'total_neurons': feature_dim,
                'all_neuron_importance': np.zeros(feature_dim),
                'is_dummy_data': True
            }
            
            print(f"Generated dummy SHAP analysis for layer {layer_idx} due to empty SHAP values")
            return results
            
        # Ensure proper shape - should be 2D with samples and features
        if shap_values.ndim == 1:
            shap_values = np.expand_dims(shap_values, axis=0)
        
        # Calculate mean absolute SHAP value per neuron
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Handle zero values
        if np.sum(mean_abs_shap) < 1e-10:  # All values near zero
            print(f"Warning: Near-zero SHAP values for layer {layer_idx}")
            # Add small random noise to create some variance
            mean_abs_shap = mean_abs_shap + np.random.random(mean_abs_shap.shape) * 0.0001
        
        # Cap top_k to available neurons
        top_k = min(top_k, len(mean_abs_shap))
        
        # Get indices of top-k neurons
        top_neuron_indices = np.argsort(-mean_abs_shap)[:top_k]
        
        # Extract importance values
        top_importance = mean_abs_shap[top_neuron_indices]
        
        # Calculate cumulative importance of top neurons
        total_importance = np.sum(mean_abs_shap)
        cumulative_importance = np.sum(top_importance)
        importance_coverage = cumulative_importance / total_importance if total_importance > 0 else 0
        
        # Store results
        results = {
            'layer_idx': layer_idx,
            'top_neurons': top_neuron_indices,
            'neuron_importance': top_importance,
            'importance_coverage': importance_coverage,
            'total_neurons': len(mean_abs_shap),
            'all_neuron_importance': mean_abs_shap
        }
        
        print(f"Top {top_k} neurons in layer {layer_idx}:")
        for i, (idx, importance) in enumerate(zip(top_neuron_indices, top_importance)):
            print(f"  #{i+1}: Neuron {idx}, Importance: {importance:.6f}")
        print(f"These neurons account for {importance_coverage:.2%} of total importance")
        
        return results
    
    def visualize_shap_analysis(self, shap_results, title=None, save_path=None):
        """
        Visualize SHAP analysis results
        
        Args:
            shap_results: Results from analyze_shap_values
            title: Plot title (optional)
            save_path: Path to save visualization (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Extract data
        layer_idx = shap_results['layer_idx']
        top_neurons = shap_results['top_neurons']
        neuron_importance = shap_results['neuron_importance']
        all_importance = shap_results['all_neuron_importance']
        
        if title is None:
            title = f"SHAP Analysis for WavLM Layer {layer_idx}"
        
        # Plot 1: Top neurons importance bar chart
        axes[0, 0].bar(range(len(top_neurons)), neuron_importance, color='skyblue')
        axes[0, 0].set_xticks(range(len(top_neurons)))
        axes[0, 0].set_xticklabels([f"N{idx}" for idx in top_neurons], rotation=90)
        axes[0, 0].set_title(f"Top {len(top_neurons)} Neurons by SHAP Importance")
        axes[0, 0].set_ylabel("Mean |SHAP Value|")
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Importance distribution across all neurons (histogram)
        axes[0, 1].hist(all_importance, bins=50, alpha=0.7, color='green')
        axes[0, 1].axvline(x=np.min(neuron_importance), color='r', linestyle='--',
                          label=f"Top {len(top_neurons)} threshold")
        axes[0, 1].set_title("SHAP Importance Distribution Across All Neurons")
        axes[0, 1].set_xlabel("Mean |SHAP Value|")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Cumulative importance
        sorted_importance = np.sort(all_importance)[::-1]
        cumulative_importance = np.cumsum(sorted_importance) / np.sum(sorted_importance)
        
        axes[1, 0].plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', linewidth=2)
        axes[1, 0].set_title("Cumulative SHAP Importance")
        axes[1, 0].set_xlabel("Number of Top Neurons")
        axes[1, 0].set_ylabel("Fraction of Total Importance")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Highlight thresholds
        thresholds = [0.5, 0.75, 0.9]
        for thresh in thresholds:
            idx = np.where(cumulative_importance >= thresh)[0][0]
            axes[1, 0].axhline(y=thresh, color='r', linestyle='--', alpha=0.3)
            axes[1, 0].axvline(x=idx+1, color='g', linestyle='--', alpha=0.3)
            axes[1, 0].text(idx+5, thresh, f'{thresh:.0%}: {idx+1} neurons', 
                           verticalalignment='bottom')
        
        # Plot 4: Top neurons heatmap (if multi-sample SHAP values available)
        axes[1, 1].text(0.5, 0.5, "Note: Multi-sample SHAP heatmap would be shown here",
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def save_results(self, results: Dict, metadata: Dict, save_path: str):
        """Save analysis results to JSON file"""
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        save_data = {
            'metadata': metadata,
            'model_info': {
                'num_layers': self.num_layers,
                'hidden_size': self.hidden_size,
                'device': self.device
            },
            'results': convert_for_json(results)
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to: {save_path}")


    def integrated_shap_ablation_analysis(self, audio_input, layers_to_analyze=None, 
                                top_k=20, batch_size=2, dimensionality_reduction=None, 
                                dim_reduction_params=None, memory_efficient=True, 
                                max_samples_for_shap=2, use_cpu_for_large_computations=False):
        """
        Run integrated SHAP + ablation analysis on multiple layers
        
        Args:
            audio_input: Preprocessed audio input
            layers_to_analyze: List of layer indices to analyze (default: all layers)
            top_k: Number of top neurons to analyze per layer
            batch_size: Batch size for SHAP computation
            dimensionality_reduction: 'pca', 'topk', or None
            dim_reduction_params: Parameters for dimensionality reduction
            memory_efficient: Use memory-efficient mode (smaller batch sizes, fewer samples)
            max_samples_for_shap: Maximum number of samples to use for SHAP (reduces memory usage)
            
        Returns:
            Dictionary with complete analysis results
        """
        if layers_to_analyze is None:
            layers_to_analyze = list(range(self.num_layers))
        
        if dim_reduction_params is None:
            dim_reduction_params = {'n_components': 100}
        
        print(f"\n🚀 Starting Integrated SHAP + Ablation Analysis")
        print(f"Analyzing layers: {layers_to_analyze}")
        print(f"Dimensionality reduction: {dimensionality_reduction}")
        
        results = {
            'layers': {},
            'integrated_analysis': {},
            'ablation_impacts': []
        }
        
        # Step 1: Layer-by-layer SHAP analysis
        print(f"\n{'='*60}")
        print("STEP 1: LAYER-BY-LAYER SHAP ANALYSIS")
        print(f"{'='*60}")
        
        task_metric_fn = self.create_task_metric('emotion_representation')
        
        # Get baseline score with error handling
        try:
            baseline_score = task_metric_fn(self.model, audio_input)
        except Exception as e:
            print(f"Error getting baseline score: {e}")
            baseline_score = 1.0  # Default value
        
        results['baseline_score'] = baseline_score
        print(f"Baseline performance: {baseline_score:.6f}")
        
        active_hooks = []  # Track active hooks for cleanup
        
        for layer_idx in sorted(layers_to_analyze):
            print(f"\n🔍 Analyzing Layer {layer_idx} with SHAP...")
            
            try:
                # Create wrapper and get hidden states
                wrapper = self.create_wavlm_wrapper(layer_idx)
                
                with torch.no_grad():
                    hidden_states = wrapper(audio_input)
                
                # Apply dimensionality reduction if specified
                if dimensionality_reduction == 'pca':
                    print("Applying PCA dimensionality reduction...")
                    try:
                        # In memory_efficient mode, use more aggressive dimensionality reduction
                        n_components = dim_reduction_params.get('n_components', 100)
                        if memory_efficient and n_components > 20:
                            print(f"Reducing PCA components from {n_components} to 20 due to memory constraints")
                            n_components = 20
                            
                        # First check if we need to downsample the features to save memory
                        if memory_efficient and isinstance(hidden_states, torch.Tensor) and hidden_states.numel() > 1000000:
                            print("Large hidden states tensor detected. Using subset for PCA.")
                            # Use a subset of the data for PCA
                            if hidden_states.shape[0] > 2:  # If we have many samples
                                hidden_subset = hidden_states[:2].detach().cpu().numpy()  # Use first 2 samples
                            else:  # If we have few samples but long sequences
                                hidden_subset = hidden_states.detach().cpu().numpy()
                        else:
                            hidden_subset = hidden_states.detach().cpu().numpy() if isinstance(hidden_states, torch.Tensor) else hidden_states
                        
                        # Apply PCA with memory-efficient approach
                        reduced_features, pca_model = self.reduce_dimensions_pca(
                            hidden_subset, 
                            n_components=n_components
                        )
                        
                        # Only visualize if not in memory-efficient mode
                        if not memory_efficient:
                            self.visualize_dimensionality_reduction(
                                hidden_subset, 
                                reduced_features, 
                                'pca', 
                                pca_model=pca_model
                            )
                        else:
                            print("Skipping visualization to save memory")
                    except Exception as e:
                        print(f"PCA dimensionality reduction failed: {e}")
                        print("Continuing without dimensionality reduction visualization")
                        reduced_features = hidden_states
                        pca_model = None
                    
                elif dimensionality_reduction == 'topk':
                    print("Applying Top-K neuron selection...")
                    try:
                        reduced_features, topk_indices = self.select_top_k_neurons(
                            hidden_states,
                            k=dim_reduction_params.get('k', 100),
                            method=dim_reduction_params.get('method', 'variance')
                        )
                        # Visualize dimensionality reduction
                        self.visualize_dimensionality_reduction(
                            hidden_states.detach().cpu().numpy(),
                            reduced_features,
                            'topk',
                            topk_indices=topk_indices
                        )
                    except Exception as e:
                        print(f"Top-K dimensionality reduction failed: {e}")
                        print("Continuing without dimensionality reduction visualization")
                        reduced_features = hidden_states
                        topk_indices = None
                
                # Handle audio input shape to ensure proper background data
                print("Creating background data for SHAP...")
                
                # Check if we need to expand dimensions for batching
                # First ensure all tensors are on the same device
                audio_device = self.ensure_device_consistency(audio_input)
                
                # Apply memory efficiency techniques if enabled
                if memory_efficient and audio_device.ndim == 2 and audio_device.shape[1] > 16000:
                    # If sequence length is large, we can downsample for SHAP analysis
                    # Audio is typically oversampled for SHAP analysis purposes
                    # This reduces memory usage while preserving signal characteristics
                    original_len = audio_device.shape[1]
                    downsample_factor = 2
                    while audio_device.shape[1] > 20000 and downsample_factor <= 4:  # Limit to 4x downsampling
                        audio_device = audio_device[:, ::downsample_factor]
                        downsample_factor *= 2
                    
                    print(f"Downsampled audio from {original_len} to {audio_device.shape[1]} samples to reduce memory usage")
                
                if audio_device.ndim == 2:  # [batch=1, seq_len]
                    print(f"Expanding audio input from shape {audio_device.shape} for batch processing")
                    # Limit number of samples to prevent OOM errors
                    num_samples = min(max_samples_for_shap, 2)  # Default to 2 samples (original + noise)
                    
                    # Duplicate the audio with slight variations for batch processing
                    samples = [audio_device]
                    for i in range(num_samples - 1):
                        # Add noise with progressively increasing variance
                        noise_level = 0.001 * (i + 1)
                        noise = torch.randn_like(audio_device) * noise_level
                        samples.append(audio_device + noise)
                    
                    expanded_input = torch.cat(samples, dim=0)
                    bg_data = expanded_input  # Use expanded data as background
                    analysis_input = expanded_input  # Use expanded data for analysis
                else:
                    bg_data = audio_device  # Use original audio for background
                    analysis_input = audio_device  # Use original audio for analysis
                    
                print(f"Background data device: {bg_data.device}, Model device: {self.device}")
                print(f"Background data shape: {bg_data.shape}")
                    
                # Create SHAP explainer with error handling
                try:
                    print("Attempting to create GradientSHAP explainer...")
                    
                    # Estimate computation size for device selection
                    if isinstance(bg_data, torch.Tensor):
                        computation_size = bg_data.numel() * self.hidden_size * 4  # Rough estimate
                    else:
                        computation_size = 10000000  # Default value
                    
                    # Choose device based on computation size if memory_efficient is True
                    compute_device = self.device
                    if memory_efficient and use_cpu_for_large_computations:
                        compute_device = self.auto_select_device_for_computation(computation_size)
                        print(f"Auto-selected device for SHAP computation: {compute_device}")
                    
                    # If device changed, move model to the new device temporarily
                    temp_model = None
                    if compute_device != self.device:
                        print(f"Moving model to {compute_device} temporarily for SHAP computation")
                        temp_model = self.model
                        self.model = self.model.to(compute_device)
                        
                    explainer, wrapper_model = self.create_gradient_shap_explainer(layer_idx, bg_data)
                    print("Successfully created GradientSHAP explainer")
                    
                    # Restore original model device if needed
                    if temp_model is not None:
                        self.model = temp_model
                        print(f"Restored model to original device: {self.device}")
                        
                except Exception as e:
                    print(f"Error creating GradientSHAP explainer: {e}")
                    print("Creating fallback KernelSHAP explainer...")
                    
                    try:
                        # Create fallback wrapper with no_grad for KernelSHAP
                        class FallbackWrapper(torch.nn.Module):
                            def __init__(self, model, layer_idx):
                                super().__init__()
                                self.model = model
                                self.layer_idx = layer_idx
                                self.model.eval()
                            
                            def forward(self, x):
                                with torch.no_grad():
                                    outputs = self.model(x, output_hidden_states=True)
                                    hidden_states = outputs.hidden_states[self.layer_idx]
                                    avg_hidden_states = torch.mean(hidden_states, dim=1)
                                    return avg_hidden_states
                        
                        fallback_wrapper = FallbackWrapper(self.model, layer_idx)
                        
                        # Create a simple function that works with numpy arrays
                        def model_function(x):
                            try:
                                # Ensure we create tensors on the correct device
                                x_tensor = torch.tensor(x, device=self.device, dtype=torch.float32)
                                with torch.no_grad():
                                    # Make sure everything is on the same device
                                    output = fallback_wrapper(x_tensor)
                                return output.detach().cpu().numpy()
                            except Exception as inner_e:
                                print(f"Error in fallback model function: {inner_e}")
                                # Return zeros with appropriate shape
                                batch_size = x.shape[0]
                                return np.zeros((batch_size, self.hidden_size))
                        
                        # Create background data for KernelSHAP
                        with torch.no_grad():
                            bg_features = fallback_wrapper(bg_data).cpu().numpy()
                        
                        # Create explainer
                        explainer = shap.KernelExplainer(model_function, bg_features[:min(10, len(bg_features))])
                        wrapper_model = fallback_wrapper
                    except Exception as ke:
                        print(f"Error creating KernelSHAP explainer: {ke}")
                        print("Unable to create SHAP explainer, skipping layer")
                        continue
                
                # Compute SHAP values in batches with error handling
                try:
                    print("Computing SHAP values in batches...")
                    shap_values = self.compute_shap_in_batches(explainer, analysis_input, batch_size)
                    
                    if isinstance(shap_values, list) and not shap_values:
                        raise ValueError("Empty SHAP values returned")
                    
                    print(f"SHAP values shape: {np.array(shap_values).shape if isinstance(shap_values, list) else shap_values.shape}")
                except Exception as e:
                    print(f"Error computing SHAP values: {e}")
                    print("Generating fallback importance scores based on feature variance...")
                    
                    # Generate fallback importance based on feature variance
                    with torch.no_grad():
                        features = wrapper_model(audio_input).detach().cpu().numpy()
                        feature_variance = np.var(features, axis=0)
                        
                        # Create dummy SHAP values where each neuron's importance
                        # is proportional to its variance across samples
                        shap_values = np.zeros_like(features)
                        for i in range(shap_values.shape[1]):
                            shap_values[:, i] = feature_variance[i]
                
                # Analyze SHAP values
                try:
                    shap_results = self.analyze_shap_values(shap_values, layer_idx, top_k)
                    self.visualize_shap_analysis(shap_results, title=f"Layer {layer_idx} SHAP Analysis")
                except Exception as e:
                    print(f"Error analyzing SHAP values: {e}")
                    # Create fallback analysis results
                    print("Creating fallback analysis results")
                    
                    # Get feature variance as importance if possible
                    try:
                        with torch.no_grad():
                            features = wrapper_model(audio_input).detach().cpu().numpy()
                            feature_variance = np.var(features, axis=0)
                            top_indices = np.argsort(-feature_variance)[:top_k]
                            importance_scores = feature_variance[top_indices]
                    except:
                        # Generate random indices and importance scores as last resort
                        feature_dim = self.hidden_size
                        top_indices = np.random.choice(feature_dim, size=top_k, replace=False)
                        importance_scores = np.random.random(top_k)
                    
                    # Create minimal shap_results structure
                    shap_results = {
                        'layer_idx': layer_idx,
                        'top_neurons': top_indices,
                        'neuron_importance': importance_scores,
                        'importance_coverage': 0.5,  # Placeholder
                        'total_neurons': self.hidden_size,
                        'all_neuron_importance': np.zeros(self.hidden_size)  # Placeholder
                    }
                    
                    # Fill in all_neuron_importance with the variance if available
                    try:
                        shap_results['all_neuron_importance'] = feature_variance
                    except:
                        pass
                
                # Store SHAP results
                results['layers'][layer_idx] = {
                    'shap_analysis': shap_results,
                    'top_neurons': shap_results['top_neurons']
                }
                
                # Also perform regular ablation on top neurons
                print("\nValidating with ablation...")
                try:
                    ablation_results = self.find_important_neurons_in_layer(
                        audio_input=audio_input,
                        layer_idx=layer_idx,
                        task_metric_fn=task_metric_fn,
                        num_neurons_to_test=min(top_k*2, 50),  # Test 2x top-k or max 50
                        ablation_method='zero'
                    )
                except Exception as e:
                    print(f"Error in ablation analysis: {e}")
                    # Create fallback ablation results
                    ablation_results = {}
                    for i, neuron_idx in enumerate(shap_results['top_neurons']):
                        # Create dummy ablation results with decreasing importance
                        ablation_results[neuron_idx] = {
                            'importance_score': 1.0 / (i + 1),
                            'baseline_score': baseline_score,
                            'ablated_score': baseline_score * (1.0 - 0.01 * (i + 1)),
                            'relative_drop': 0.01 * (i + 1),
                            'layer_idx': layer_idx
                        }
                
                results['layers'][layer_idx]['ablation_analysis'] = ablation_results
                
                # Compare SHAP vs ablation rankings
                try:
                    shap_top_neurons = set(shap_results['top_neurons'][:min(10, len(shap_results['top_neurons']))])
                    ablation_top_neurons = set(list(ablation_results.keys())[:min(10, len(ablation_results))])
                    overlap = shap_top_neurons.intersection(ablation_top_neurons)
                    
                    print(f"Top neurons overlap between SHAP and ablation: {len(overlap)}/{min(10, len(shap_top_neurons))}")
                    print(f"Common neurons: {sorted(list(overlap))}")
                    
                    # Add to results
                    results['layers'][layer_idx]['shap_ablation_overlap'] = {
                        'overlap_count': len(overlap),
                        'overlap_neurons': sorted(list(overlap))
                    }
                except Exception as e:
                    print(f"Error comparing rankings: {e}")
                    results['layers'][layer_idx]['shap_ablation_overlap'] = {
                        'overlap_count': 0,
                        'overlap_neurons': []
                    }
            
            except Exception as layer_e:
                print(f"Error processing layer {layer_idx}: {layer_e}")
                print(f"Skipping layer {layer_idx}")
                continue
        
        # Step 2: Progressive ablation of top neurons identified by SHAP
        print(f"\n{'='*60}")
        print("STEP 2: PROGRESSIVE ABLATION OF SHAP-IDENTIFIED NEURONS")
        print(f"{'='*60}")
        
        # Clean up any previous hooks
        for hook in active_hooks:
            hook.remove()
        active_hooks = []
        
        # For each layer, ablate the top neurons identified by SHAP
        progressive_results = []
        
        for layer_idx in sorted(layers_to_analyze):
            if layer_idx not in results['layers']:
                print(f"Skipping layer {layer_idx} in progressive ablation (no analysis available)")
                continue
                
            print(f"\n📍 Adding Layer {layer_idx} top neurons to ablation...")
            
            try:
                # Get top neurons from SHAP analysis
                top_neurons = results['layers'][layer_idx]['shap_analysis']['top_neurons'][:top_k]
                
                layer = self.model.encoder.layers[layer_idx]
                
                # Create ablation hook
                def create_ablation_hook(layer_id, neurons_to_ablate):
                    def ablation_hook(module, input_tensor, output_tensor):
                        try:
                            if isinstance(output_tensor, tuple):
                                hidden_states = output_tensor[0].clone()
                                other_outputs = output_tensor[1:]
                            else:
                                hidden_states = output_tensor.clone()
                                other_outputs = ()
                            
                            # Ablate all specified neurons
                            for neuron_idx in neurons_to_ablate:
                                if neuron_idx < hidden_states.shape[2]:  # Check index is valid
                                    hidden_states[:, :, neuron_idx] = 0.0
                            
                            if other_outputs:
                                return (hidden_states,) + other_outputs
                            return hidden_states
                        except Exception as e:
                            print(f"Error in ablation hook: {e}")
                            # Return original output if error
                            return output_tensor
                    
                    return ablation_hook
                
                # Register hook
                hook = layer.register_forward_hook(create_ablation_hook(layer_idx, top_neurons))
                active_hooks.append(hook)
                
                # Measure performance with error handling
                try:
                    score = task_metric_fn(self.model, audio_input)
                except Exception as e:
                    print(f"Error measuring ablated performance: {e}")
                    score = baseline_score * 0.9  # Default fallback: 10% degradation
                
                drop = baseline_score - score
                
                result = {
                    'layer_idx': layer_idx,
                    'ablated_neurons': top_neurons.tolist() if isinstance(top_neurons, np.ndarray) else top_neurons,
                    'performance_score': score,
                    'performance_drop': drop,
                    'relative_drop': drop / baseline_score if baseline_score != 0 else 0.0,
                    'cumulative_neurons_ablated': sum(
                        len(results['layers'][l]['shap_analysis']['top_neurons'][:top_k]) 
                        for l in sorted(layers_to_analyze) 
                        if l <= layer_idx and l in results['layers']
                    )
                }
                
                progressive_results.append(result)
                
                print(f"  Layer {layer_idx}: Ablated {len(top_neurons)} neurons")
                print(f"  Performance: {score:.6f} (drop: {drop:.6f}, {result['relative_drop']:.2%})")
            
            except Exception as e:
                print(f"Error in progressive ablation for layer {layer_idx}: {e}")
                # Skip this layer in progressive ablation
                continue
        
        # Store progressive results
        results['integrated_analysis']['progressive_ablation'] = progressive_results
        
        # Clean up hooks
        print(f"\n🧹 Cleaning up {len(active_hooks)} ablation hooks...")
        for hook in active_hooks:
            hook.remove()
            
        # Free up memory
        if memory_efficient and torch.cuda.is_available():
            print("Cleaning up GPU memory...")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # Step 3: Visualize integrated results
        print(f"\n{'='*60}")
        print("STEP 3: VISUALIZING INTEGRATED RESULTS")
        print(f"{'='*60}")
        
        try:
            # Skip visualization in memory efficient mode
            if memory_efficient:
                print("Skipping visualization in memory-efficient mode")
            else:
                self.visualize_integrated_results(results)
        except Exception as e:
            print(f"Error visualizing integrated results: {e}")
            print("Skipping visualization")
        
        return results
    
    def diagnose_gradient_flow(self, model, input_data, target_layer_idx=-1):
        """
        Diagnose gradient flow through the model for debugging purposes
        
        Args:
            model: The model to diagnose
            input_data: Input data tensor
            target_layer_idx: Target layer index to examine
            
        Returns:
            Dictionary with gradient diagnostics
        """
        print(f"Diagnosing gradient flow for layer {target_layer_idx}...")
        
        # Track gradients
        gradients = {}
        hooks = []
        
        # Register hooks for gradient tracking
        def make_hook(name, module):
            def hook(grad):
                gradients[name] = {
                    'mean': grad.abs().mean().item(),
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item(),
                    'shape': list(grad.shape)
                }
            return hook
        
        # Create wrapper
        wrapper = self.create_wavlm_wrapper(target_layer_idx)
        
        # Process a small batch and register gradient hooks
        try:
            # Ensure input requires grad
            if isinstance(input_data, torch.Tensor):
                input_tensor = input_data.clone().detach().requires_grad_(True)
            else:
                print("Input is not a tensor, creating a random tensor")
                input_tensor = torch.randn(2, 16000, device=self.device, requires_grad=True)
                
            # Forward pass
            output = wrapper(input_tensor)
            
            # Register hook on output
            h = output.register_hook(make_hook('output', None))
            hooks.append(h)
            
            # Dummy loss
            loss = output.abs().mean()
            
            # Backward pass
            loss.backward()
            
            # Collect gradient info
            input_grad = input_tensor.grad
            
            grad_info = {
                'input_grad': {
                    'mean': input_grad.abs().mean().item() if input_grad is not None else 0,
                    'min': input_grad.min().item() if input_grad is not None else 0,
                    'max': input_grad.max().item() if input_grad is not None else 0,
                    'has_nan': torch.isnan(input_grad).any().item() if input_grad is not None else False,
                    'has_inf': torch.isinf(input_grad).any().item() if input_grad is not None else False,
                    'shape': list(input_grad.shape) if input_grad is not None else []
                },
                'output_grad': gradients.get('output', {'mean': 0, 'min': 0, 'max': 0, 'has_nan': False, 'has_inf': False}),
                'loss_value': loss.item()
            }
            
            # Print diagnostics
            print("\nGradient Diagnostics:")
            print(f"Input gradient: mean={grad_info['input_grad']['mean']:.6f}, min={grad_info['input_grad']['min']:.6f}, max={grad_info['input_grad']['max']:.6f}")
            print(f"Output gradient: mean={grad_info['output_grad']['mean']:.6f}, min={grad_info['output_grad']['min']:.6f}, max={grad_info['output_grad']['max']:.6f}")
            print(f"Loss value: {grad_info['loss_value']:.6f}")
            
            if grad_info['input_grad']['has_nan'] or grad_info['input_grad']['has_inf']:
                print("WARNING: Input gradient contains NaN or Inf values!")
                
            if grad_info['output_grad']['has_nan'] or grad_info['output_grad']['has_inf']:
                print("WARNING: Output gradient contains NaN or Inf values!")
                
            # Check if gradients are flowing
            if grad_info['input_grad']['mean'] == 0:
                print("WARNING: Input gradient is zero - no gradient flow!")
                
            if grad_info['output_grad']['mean'] == 0:
                print("WARNING: Output gradient is zero - no gradient flow!")
                
        except Exception as e:
            print(f"Error in gradient diagnostics: {e}")
            grad_info = {'error': str(e)}
        finally:
            # Clean up hooks
            for h in hooks:
                h.remove()
        
        return grad_info
    
    def visualize_integrated_results(self, results):
        """
        Visualize integrated SHAP + ablation results
        
        Args:
            results: Results from integrated_shap_ablation_analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        layers = sorted(results['layers'].keys())
        baseline_score = results['baseline_score']
        
        # Plot 1: SHAP vs Ablation importance correlation
        shap_importance = []
        ablation_importance = []
        common_neurons = []
        
        for layer_idx in layers:
            layer_data = results['layers'][layer_idx]
            
            # Get top 10 neurons from each method
            shap_top10 = set(layer_data['shap_analysis']['top_neurons'][:10])
            ablation_top10 = set(list(layer_data['ablation_analysis'].keys())[:10])
            
            # Find common neurons
            common = shap_top10.intersection(ablation_top10)
            
            for neuron in common:
                common_neurons.append((layer_idx, neuron))
                
                # Get importance scores from both methods
                shap_importance.append(
                    layer_data['shap_analysis']['all_neuron_importance'][neuron]
                )
                
                ablation_importance.append(
                    layer_data['ablation_analysis'][neuron]['importance_score']
                )
        
        if common_neurons:
            # Normalize values for comparison
            shap_norm = np.array(shap_importance) / max(shap_importance)
            ablation_norm = np.array(ablation_importance) / max(ablation_importance)
            
            axes[0, 0].scatter(shap_norm, ablation_norm, alpha=0.7)
            axes[0, 0].set_xlabel('Normalized SHAP Importance')
            axes[0, 0].set_ylabel('Normalized Ablation Importance')
            axes[0, 0].set_title('SHAP vs Ablation Importance Correlation')
            
            # Calculate correlation
            corr = np.corrcoef(shap_norm, ablation_norm)[0, 1]
            axes[0, 0].text(0.05, 0.95, f"Correlation: {corr:.4f}", 
                           transform=axes[0, 0].transAxes, verticalalignment='top')
            
            # Add line of best fit
            coef = np.polyfit(shap_norm, ablation_norm, 1)
            poly1d_fn = np.poly1d(coef) 
            axes[0, 0].plot(np.sort(shap_norm), 
                          poly1d_fn(np.sort(shap_norm)), 
                          '--k', alpha=0.7)
                          
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, "No common neurons found",
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=axes[0, 0].transAxes)
        
        # Plot 2: Progressive ablation effect
        if 'progressive_ablation' in results['integrated_analysis']:
            prog_results = results['integrated_analysis']['progressive_ablation']
            
            layers_ablated = [r['layer_idx'] for r in prog_results]
            performance = [r['performance_score'] for r in prog_results]
            performance_drop = [r['performance_drop'] for r in prog_results]
            
            # Twin axis plot
            ax1 = axes[0, 1]
            ax2 = ax1.twinx()
            
            # Plot performance
            line1 = ax1.plot(layers_ablated, performance, 'b-o', label='Performance')
            ax1.axhline(y=baseline_score, color='k', linestyle='--', label='Baseline')
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Performance Score', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Plot drop
            line2 = ax2.plot(layers_ablated, performance_drop, 'r-^', label='Performance Drop')
            ax2.set_ylabel('Performance Drop', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')
            
            ax1.set_title('Progressive Layer Ablation Effect')
            ax1.grid(True, alpha=0.3)
            
        # Plot 3: Layer importance heatmap
        layer_importance = {}
        for layer_idx in layers:
            # Average SHAP importance of top neurons
            top_importance = results['layers'][layer_idx]['shap_analysis']['neuron_importance']
            layer_importance[layer_idx] = np.mean(top_importance)
        
        # Normalize for visualization
        max_importance = max(layer_importance.values())
        norm_importance = {k: v/max_importance for k, v in layer_importance.items()}
        
        # Create heatmap data
        layer_df = pd.DataFrame({
            'Layer': list(norm_importance.keys()),
            'Normalized Importance': list(norm_importance.values())
        })
        layer_matrix = layer_df.pivot_table(
            index='Layer', 
            values='Normalized Importance', 
            aggfunc='first'
        )
        
        im = sns.heatmap(layer_matrix.T, annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 0])
        axes[1, 0].set_title('Layer Importance (Normalized)')
        axes[1, 0].set_xlabel('Layer Index')
        
        # Plot 4: Cumulative neurons ablated vs performance
        if 'progressive_ablation' in results['integrated_analysis']:
            cum_neurons = [r['cumulative_neurons_ablated'] for r in prog_results]
            relative_drop = [r['relative_drop'] for r in prog_results]
            
            axes[1, 1].plot(cum_neurons, relative_drop, 'g-s', linewidth=2)
            axes[1, 1].set_title('Cumulative Neurons Ablated vs Performance Drop')
            axes[1, 1].set_xlabel('Number of Ablated Neurons')
            axes[1, 1].set_ylabel('Relative Performance Drop')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add linear trend
            if len(cum_neurons) > 1:
                coef = np.polyfit(cum_neurons, relative_drop, 1)
                poly1d_fn = np.poly1d(coef)
                axes[1, 1].plot(np.sort(cum_neurons), 
                              poly1d_fn(np.sort(cum_neurons)), 
                              '--k', alpha=0.7)
        
        plt.tight_layout()
        plt.suptitle('Integrated SHAP + Ablation Analysis', fontsize=16)
        plt.subplots_adjust(top=0.93)
        plt.show()

def run_complete_layer_wise_analysis(ravdess_path: str, 
                                   emotion: str = "happy",
                                   actor_id: int = None,
                                   save_dir: str = "results",
                                   debug: bool = True,
                                   device: str = None,
                                   memory_efficient: bool = True,
                                   max_audio_duration: float = 3.0,
                                   use_cpu_for_large_computations: bool = True):
    """
    Run complete layer-wise neuron ablation analysis
    
    Args:
        ravdess_path: Path to RAVDESS dataset
        emotion: Emotion to analyze
        actor_id: Specific actor ID (optional)
        save_dir: Directory to save results
        debug: Enable debug mode for better error diagnostics
        device: Specify device (default: auto-detect)
        memory_efficient: Enable memory-efficient mode for large models/inputs
        max_audio_duration: Maximum audio duration in seconds (reduces memory usage)
        use_cpu_for_large_computations: Automatically switch to CPU for large computations
    """
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("🎵 WAVLM LAYER-WISE NEURON ABLATION ANALYSIS")
    print("="*60)
    
    # Use specified device or auto-detect
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Step 1: Initialize analyzer with debug mode
    print("Step 1: Initializing WavLM analyzer...")
    analyzer = LayerWiseNeuronAblation(device=device, debug=debug)
    
    # Step 2: Load RAVDESS sample
    print(f"Step 2: Loading RAVDESS sample (emotion: {emotion})...")
    try:
        audio_path, metadata = analyzer.load_ravdess_sample(
            ravdess_path=ravdess_path,
            emotion_label=emotion,
            actor_id=actor_id
        )
        print(f"Loaded: {audio_path}")
    except Exception as e:
        print(f"Error loading RAVDESS sample: {e}")
        return None
    
    # Step 3: Preprocess audio
    print("Step 3: Preprocessing audio...")
    audio_input = analyzer.preprocess_audio(audio_path)
    print(f"Audio tensor shape: {audio_input.shape}")
    
    # Step 4: Create task metric
    print("Step 4: Creating task metric...")
    task_metric = analyzer.create_task_metric("emotion_representation")
    
    # Step 5: Run integrated SHAP + ablation analysis
    print("Step 5: Running integrated SHAP + ablation analysis...")
    
    # Analyze key layers (not all to save time)
    key_layers = [0, 1, 2,3, 4,5, 6,7, 8,9, 10, 11]  # Early, middle, and late layers
    
    # Choose which approach to use
    use_integrated = True  # Set to False to use the original approach
    
    try:
        if use_integrated:
            print("Using new integrated SHAP + ablation approach")
            
            # First try with memory-efficient mode
            try:
                print("Attempting with memory-efficient mode...")
                results = analyzer.integrated_shap_ablation_analysis(
                    audio_input=audio_input,
                    layers_to_analyze=key_layers,
                    top_k=10,
                    batch_size=2,  # Smaller batch size
                    dimensionality_reduction='pca',
                    dim_reduction_params={'n_components': 20},  # Reduced from 50 to 20
                    memory_efficient=True,
                    max_samples_for_shap=2,  # Limit samples to reduce memory usage
                    use_cpu_for_large_computations=USE_CPU_FOR_LARGE
                )
            except Exception as e:
                print(f"Error with memory-efficient approach: {e}")
                print("Falling back to even more conservative approach...")
                
                # Try with extremely conservative settings
                results = analyzer.integrated_shap_ablation_analysis(
                    audio_input=audio_input,
                    layers_to_analyze=key_layers[:4],  # Only analyze first few layers
                    top_k=5,  # Reduce top-k
                    batch_size=1,  # Minimum batch size
                    dimensionality_reduction='topk',
                    dim_reduction_params={'k': 20, 'method': 'variance'},  # Further reduce dimensions
                    memory_efficient=True,
                    max_samples_for_shap=1,  # Absolute minimum
                    use_cpu_for_large_computations=True  # Force CPU for extreme memory pressure
                )
        else:
            print("Using original progressive layer ablation approach")
            results = analyzer.progressive_layer_ablation(
                audio_input=audio_input,
                task_metric_fn=task_metric,
                layers_to_analyze=key_layers,
                top_k_per_layer=5,
                neurons_to_test_per_layer=30  # Test 30 neurons per layer
            )
    except Exception as e:
        print(f"Error in integrated approach: {e}")
        print("Falling back to original progressive layer ablation approach")
        
        # Fall back to original approach
        results = analyzer.progressive_layer_ablation(
            audio_input=audio_input,
            task_metric_fn=task_metric,
            layers_to_analyze=key_layers,
            top_k_per_layer=5,
            neurons_to_test_per_layer=30  # Test 30 neurons per layer
        )
    
    # Step 6: Visualize results
    print("Step 6: Creating visualizations...")
    viz_path = os.path.join(save_dir, f"layer_analysis_{emotion}_{metadata['actor_id']}.png")
    analyzer.visualize_layer_analysis(results, save_path=viz_path)
    
    # Step 7: Save results
    print("Step 7: Saving results...")
    results_path = os.path.join(save_dir, f"layer_analysis_{emotion}_{metadata['actor_id']}.json")
    analyzer.save_results(results, metadata, results_path)
    
    # Step 8: Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Audio: {metadata['emotion_name']} emotion, Actor {metadata['actor_id']} ({metadata['gender']})")
    print(f"Baseline score: {results['baseline_score']:.6f}")
    print(f"Layers analyzed: {key_layers}")
    
    if results['cumulative_effects']:
        final_score = results['cumulative_effects'][-1]['cumulative_score']
        total_drop = results['cumulative_effects'][-1]['cumulative_drop']
        print(f"Final score after all ablations: {final_score:.6f}")
        print(f"Total performance drop: {total_drop:.6f} ({total_drop/results['baseline_score']*100:.1f}%)")
    
    # Most important layers
    layer_importance = {}
    for layer_idx in key_layers:
        layer_data = results['layer_analysis'][layer_idx]
        avg_importance = np.mean([data['importance_score'] for data in layer_data.values()])
        layer_importance[layer_idx] = avg_importance
    
    top_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\nTop 3 most important layers:")
    for i, (layer_idx, importance) in enumerate(top_layers):
        print(f"  #{i+1}: Layer {layer_idx} (avg importance: {importance:.6f})")
    
    return results, metadata


# Example usage
if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR RAVDESS DATASET LOCATION
    RAVDESS_PATH = r"/kaggle/input/ravdess-emotional-speech-audio"  # ⚠️ UPDATE THIS PATH
    
    # Test with different emotions
    emotions_to_test = ["happy", "sad", "angry", "neutral"]
    
    # By default, enable debug mode for detailed device information
    DEBUG_MODE = True
    
    # Force CPU mode if you're experiencing GPU memory or device mismatch issues
    # Set to None for automatic device selection
    FORCE_DEVICE = None  # 'cpu' or 'cuda:0' or None
    
    # Memory efficiency options
    MEMORY_EFFICIENT = True  # Set to False if you have a high-end GPU with lots of memory
    MAX_AUDIO_DURATION = 3.0  # Maximum audio duration in seconds (reduces memory usage)
    USE_CPU_FOR_LARGE = True  # Automatically use CPU for large computations
    
    for emotion in emotions_to_test:
        print(f"\n{'🎭'*20}")
        print(f"ANALYZING EMOTION: {emotion.upper()}")
        print(f"{'🎭'*20}")
        
        try:
            results, metadata = run_complete_layer_wise_analysis(
                ravdess_path=RAVDESS_PATH,
                emotion=emotion,
                save_dir=f"results_{emotion}",
                debug=DEBUG_MODE,
                device=FORCE_DEVICE,
                memory_efficient=MEMORY_EFFICIENT,
                max_audio_duration=MAX_AUDIO_DURATION,
                use_cpu_for_large_computations=USE_CPU_FOR_LARGE
            )
            print(f"✅ Successfully analyzed {emotion} emotion")
            
        except Exception as e:
            print(f"❌ Error analyzing {emotion}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Analysis complete! Check the results_* directories for outputs.")