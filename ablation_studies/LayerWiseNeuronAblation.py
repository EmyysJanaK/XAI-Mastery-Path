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
    
    def __init__(self, model_name="microsoft/wavlm-base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize WavLM model with proper feature extractor"""
        self.device = device
        
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
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:
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
        
        # ✅ Use feature extractor instead of processor
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=target_sr, 
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
        
        # Ensure we don't try to extract more components than possible
        max_components = min(features_np.shape[0], features_np.shape[1])
        safe_n_components = min(n_components, max_components - 1)
        
        if safe_n_components < n_components:
            print(f"Warning: Reducing n_components from {n_components} to {safe_n_components} due to data shape constraints")
        
        # Apply PCA
        pca = PCA(n_components=safe_n_components)
        reduced_features = pca.fit_transform(features_np)
        
        print(f"Original features: {features_np.shape}, Reduced: {reduced_features.shape}")
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        return reduced_features, pca
    
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
        # Define wrapper class
        class WavLMWrapper(torch.nn.Module):
            def __init__(self, model, layer_idx):
                super().__init__()
                self.model = model
                self.layer_idx = layer_idx
                self.model.eval()
            
            def forward(self, x):
                with torch.no_grad():
                    outputs = self.model(x, output_hidden_states=True)
                    
                # Extract hidden states from the specified layer
                hidden_states = outputs.hidden_states[self.layer_idx]
                
                # Average over sequence length to get a fixed-size representation
                avg_hidden_states = torch.mean(hidden_states, dim=1)
                
                return avg_hidden_states
        
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
        
        # Create background data if not provided
        if background_data is None:
            # Generate random background data (10 samples)
            print("No background data provided, generating random samples...")
            background_data = torch.randn(10, 16000, device=self.device)
        
        # Ensure we have at least 2 samples for background data
        # as GradientSHAP requires more than one background sample
        if isinstance(background_data, torch.Tensor):
            if background_data.ndim == 2:  # [batch=1, seq_len]
                if background_data.shape[0] == 1:
                    print("Expanding single background sample to multiple samples...")
                    background_data = torch.cat([
                        background_data,
                        background_data + torch.randn_like(background_data) * 0.001
                    ], dim=0)
            
            print(f"Background data shape: {background_data.shape}")
        
        # Create explainer
        try:
            start_time = time.time()
            explainer = shap.GradientExplainer(model_wrapper, background_data)
            print(f"Explainer creation took {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error creating GradientExplainer: {e}")
            print("Falling back to simpler background data...")
            # Generate simpler background data as fallback
            random_bg = torch.randn(5, background_data.shape[1], device=self.device)
            explainer = shap.GradientExplainer(model_wrapper, random_bg)
            print("Created explainer with fallback background data")
        
        return explainer, model_wrapper
    
    def compute_shap_in_batches(self, explainer, data, batch_size=8):
        """
        Compute SHAP values in batches to reduce memory usage and improve speed
        
        Args:
            explainer: SHAP explainer
            data: Dataset to analyze
            batch_size: Batch size for processing
            
        Returns:
            SHAP values for all samples
        """
        print(f"Computing SHAP values in batches (batch_size={batch_size})...")
        
        # Get data length
        if isinstance(data, torch.Tensor):
            data_len = data.shape[0]
        else:
            data_len = len(data)
        
        # Adjust batch size for small datasets
        batch_size = min(batch_size, data_len)
        if batch_size == 0:
            print("Error: No data to process")
            return np.array([])
        
        print(f"Processing {data_len} samples with adjusted batch size: {batch_size}")
        
        # Process in batches
        all_shap_values = []
        start_time = time.time()
        
        try:
            for i in tqdm(range(0, data_len, batch_size), desc="SHAP Batch Processing"):
                # Get batch
                batch_end = min(i + batch_size, data_len)
                if isinstance(data, torch.Tensor):
                    batch = data[i:batch_end]
                else:
                    batch = data[i:batch_end]
                
                try:
                    # Compute SHAP values for batch
                    batch_shap_values = explainer.shap_values(batch)
                    
                    # Store results
                    if isinstance(batch_shap_values, list):
                        all_shap_values.extend(batch_shap_values)
                    else:
                        # If a single array is returned
                        all_shap_values.append(batch_shap_values)
                except Exception as e:
                    print(f"Error processing batch {i}:{batch_end}: {e}")
                    # Create dummy SHAP values as placeholder
                    if isinstance(batch, torch.Tensor) and batch.ndim >= 2:
                        dummy_shape = batch.shape[0]
                        feature_dim = explainer.expected_value.shape[0] if hasattr(explainer.expected_value, 'shape') else 768
                        print(f"Creating dummy SHAP values with shape [{dummy_shape}, {feature_dim}]")
                        dummy_values = np.zeros((dummy_shape, feature_dim))
                        all_shap_values.extend([dummy_values[j] for j in range(dummy_shape)])
        except Exception as e:
            print(f"Error in batch processing: {e}")
            if not all_shap_values:
                print("Attempting to compute SHAP values without batching...")
                try:
                    # Try without batching
                    all_shap_values = explainer.shap_values(data)
                    if not isinstance(all_shap_values, list):
                        all_shap_values = [all_shap_values]
                except Exception as inner_e:
                    print(f"Error in non-batch processing: {inner_e}")
                    # Create dummy data
                    feature_dim = 768  # Default
                    if hasattr(data, 'shape') and len(data.shape) >= 2:
                        dummy_shape = data.shape[0]
                    else:
                        dummy_shape = 1
                    print(f"Creating dummy SHAP values with shape [{dummy_shape}, {feature_dim}]")
                    all_shap_values = [np.zeros((dummy_shape, feature_dim))]
        
        total_time = time.time() - start_time
        print(f"Batch processing completed in {total_time:.2f} seconds")
        if data_len > 0:
            print(f"Average time per sample: {total_time / data_len:.4f} seconds")
        
        # Convert to numpy array with proper handling
        if all_shap_values:
            if isinstance(all_shap_values[0], np.ndarray):
                return np.array(all_shap_values)
            else:
                return all_shap_values
        else:
            print("Warning: No SHAP values were computed")
            return np.array([])  # Return empty array
    
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
        # Calculate mean absolute SHAP value per neuron
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
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
                                top_k=20, batch_size=8, dimensionality_reduction=None, 
                                dim_reduction_params=None):
        """
        Run integrated SHAP + ablation analysis on multiple layers
        
        Args:
            audio_input: Preprocessed audio input
            layers_to_analyze: List of layer indices to analyze (default: all layers)
            top_k: Number of top neurons to analyze per layer
            batch_size: Batch size for SHAP computation
            dimensionality_reduction: 'pca', 'topk', or None
            dim_reduction_params: Parameters for dimensionality reduction
            
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
        baseline_score = task_metric_fn(self.model, audio_input)
        results['baseline_score'] = baseline_score
        print(f"Baseline performance: {baseline_score:.6f}")
        
        active_hooks = []  # Track active hooks for cleanup
        
        for layer_idx in sorted(layers_to_analyze):
            print(f"\n🔍 Analyzing Layer {layer_idx} with SHAP...")
            
            # Create wrapper and get hidden states
            wrapper = self.create_wavlm_wrapper(layer_idx)
            
            with torch.no_grad():
                hidden_states = wrapper(audio_input)
            
            # Apply dimensionality reduction if specified
            if dimensionality_reduction == 'pca':
                print("Applying PCA dimensionality reduction...")
                reduced_features, pca_model = self.reduce_dimensions_pca(
                    hidden_states, 
                    n_components=dim_reduction_params.get('n_components', 100)
                )
                # Visualize dimensionality reduction
                self.visualize_dimensionality_reduction(
                    hidden_states.detach().cpu().numpy(), 
                    reduced_features, 
                    'pca', 
                    pca_model=pca_model
                )
                # Only for visualization - we still use full features for SHAP
                
            elif dimensionality_reduction == 'topk':
                print("Applying Top-K neuron selection...")
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
                # Only for visualization - we still use full features for SHAP
                
            # Handle audio input shape to ensure proper background data
            print("Creating background data for SHAP...")
            
            # Check if we need to expand dimensions for batching
            if audio_input.ndim == 2:  # [batch=1, seq_len]
                print(f"Expanding audio input from shape {audio_input.shape} for batch processing")
                # Duplicate the audio with slight variations for batch processing
                expanded_input = torch.cat([
                    audio_input,
                    audio_input + torch.randn_like(audio_input) * 0.001
                ], dim=0)
                bg_data = expanded_input  # Use expanded data as background
                analysis_input = expanded_input  # Use expanded data for analysis
            else:
                bg_data = audio_input  # Use original audio for background
                analysis_input = audio_input  # Use original audio for analysis
                
            # Create SHAP explainer
            explainer, _ = self.create_gradient_shap_explainer(layer_idx, bg_data)
            
            # Compute SHAP values in batches
            shap_values = self.compute_shap_in_batches(explainer, analysis_input, batch_size)
            
            # Analyze SHAP values
            shap_results = self.analyze_shap_values(shap_values, layer_idx, top_k)
            self.visualize_shap_analysis(shap_results, title=f"Layer {layer_idx} SHAP Analysis")
            
            results['layers'][layer_idx] = {
                'shap_analysis': shap_results,
                'top_neurons': shap_results['top_neurons']
            }
            
            # Also perform regular ablation on top neurons
            print("\nValidating with ablation...")
            ablation_results = self.find_important_neurons_in_layer(
                audio_input=audio_input,
                layer_idx=layer_idx,
                task_metric_fn=task_metric_fn,
                num_neurons_to_test=min(top_k*2, 50),  # Test 2x top-k or max 50
                ablation_method='zero'
            )
            
            results['layers'][layer_idx]['ablation_analysis'] = ablation_results
            
            # Compare SHAP vs ablation rankings
            shap_top_neurons = set(shap_results['top_neurons'][:10])
            ablation_top_neurons = set(list(ablation_results.keys())[:10])
            overlap = shap_top_neurons.intersection(ablation_top_neurons)
            
            print(f"Top 10 neurons overlap between SHAP and ablation: {len(overlap)}/{10}")
            print(f"Common neurons: {sorted(list(overlap))}")
            
            # Add to results
            results['layers'][layer_idx]['shap_ablation_overlap'] = {
                'overlap_count': len(overlap),
                'overlap_neurons': sorted(list(overlap))
            }
        
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
            print(f"\n📍 Adding Layer {layer_idx} top neurons to ablation...")
            
            # Get top neurons from SHAP analysis
            top_neurons = results['layers'][layer_idx]['shap_analysis']['top_neurons'][:top_k]
            
            layer = self.model.encoder.layers[layer_idx]
            
            # Create ablation hook
            def create_ablation_hook(layer_id, neurons_to_ablate):
                def ablation_hook(module, input_tensor, output_tensor):
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
                
                return ablation_hook
            
            # Register hook
            hook = layer.register_forward_hook(create_ablation_hook(layer_idx, top_neurons))
            active_hooks.append(hook)
            
            # Measure performance
            score = task_metric_fn(self.model, audio_input)
            drop = baseline_score - score
            
            result = {
                'layer_idx': layer_idx,
                'ablated_neurons': top_neurons,
                'performance_score': score,
                'performance_drop': drop,
                'relative_drop': drop / baseline_score if baseline_score != 0 else 0.0,
                'cumulative_neurons_ablated': sum(len(results['layers'][l]['shap_analysis']['top_neurons'][:top_k]) 
                                               for l in sorted(layers_to_analyze) if l <= layer_idx)
            }
            
            progressive_results.append(result)
            
            print(f"  Layer {layer_idx}: Ablated {len(top_neurons)} neurons")
            print(f"  Performance: {score:.6f} (drop: {drop:.6f}, {result['relative_drop']:.2%})")
        
        # Store progressive results
        results['integrated_analysis']['progressive_ablation'] = progressive_results
        
        # Clean up hooks
        print(f"\n🧹 Cleaning up {len(active_hooks)} ablation hooks...")
        for hook in active_hooks:
            hook.remove()
        
        # Step 3: Visualize integrated results
        print(f"\n{'='*60}")
        print("STEP 3: VISUALIZING INTEGRATED RESULTS")
        print(f"{'='*60}")
        
        self.visualize_integrated_results(results)
        
        return results
    
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
                                   save_dir: str = "results"):
    """
    Run complete layer-wise neuron ablation analysis
    
    Args:
        ravdess_path: Path to RAVDESS dataset
        emotion: Emotion to analyze
        actor_id: Specific actor ID (optional)
        save_dir: Directory to save results
    """
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("🎵 WAVLM LAYER-WISE NEURON ABLATION ANALYSIS")
    print("="*60)
    
    # Step 1: Initialize analyzer
    print("Step 1: Initializing WavLM analyzer...")
    analyzer = LayerWiseNeuronAblation()
    
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
            
            # First try with PCA dimensionality reduction
            try:
                print("Attempting with PCA dimensionality reduction...")
                results = analyzer.integrated_shap_ablation_analysis(
                    audio_input=audio_input,
                    layers_to_analyze=key_layers,
                    top_k=10,
                    batch_size=4,
                    dimensionality_reduction='pca',
                    dim_reduction_params={'n_components': 50}  # Reduced from 100 to 50
                )
            except Exception as e:
                print(f"Error with PCA approach: {e}")
                print("Falling back to TopK dimensionality reduction...")
                
                # Try with TopK dimensionality reduction
                results = analyzer.integrated_shap_ablation_analysis(
                    audio_input=audio_input,
                    layers_to_analyze=key_layers,
                    top_k=10,
                    batch_size=4,
                    dimensionality_reduction='topk',
                    dim_reduction_params={'k': 50, 'method': 'variance'}
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
    
    for emotion in emotions_to_test:
        print(f"\n{'🎭'*20}")
        print(f"ANALYZING EMOTION: {emotion.upper()}")
        print(f"{'🎭'*20}")
        
        try:
            results, metadata = run_complete_layer_wise_analysis(
                ravdess_path=RAVDESS_PATH,
                emotion=emotion,
                save_dir=f"results_{emotion}"
            )
            print(f"✅ Successfully analyzed {emotion} emotion")
            
        except Exception as e:
            print(f"❌ Error analyzing {emotion}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Analysis complete! Check the results_* directories for outputs.")