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
from typing import Dict, List, Tuple, Any, Optional
import json
from tqdm import tqdm

class GradientSHAPNeuronAnalysis:
    """
    Layer-wise neuron importance analysis using gradient-based SHAP for WavLM on RAVDESS dataset
    """
    
    def __init__(self, model_name="microsoft/wavlm-base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize WavLM model with proper feature extractor"""
        self.device = device
        
        # Use FeatureExtractor instead of Processor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(device)
        self.model.eval()  # Ensure model is in evaluation mode
        
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
        
        # Use feature extractor
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
    
    def get_layer_activation_gradient(self, 
                                    audio_input: torch.Tensor,
                                    layer_idx: int,
                                    task_type: str = "emotion_representation"):
        """
        Get activation and gradient for a specific layer
        
        Args:
            audio_input: Preprocessed audio tensor
            layer_idx: Layer index to analyze
            task_type: Type of task for loss function
            
        Returns:
            Tuple of (activation, gradient) tensors
        """
        # Prepare model and input
        audio_input = audio_input.clone().detach().requires_grad_(True)
        
        # Get the target layer
        target_layer = self.model.encoder.layers[layer_idx]
        
        # Create containers for activation and gradient
        activation = None
        gradient = None
        
        # Define forward hook to capture activation
        def forward_hook(module, input_, output):
            nonlocal activation
            if isinstance(output, tuple):
                activation = output[0].detach()
            else:
                activation = output.detach()
            
        # Define backward hook to capture gradient
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradient
            if isinstance(grad_output, tuple):
                gradient = grad_output[0].detach()
            else:
                gradient = grad_output.detach()
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        try:
            # Forward pass
            outputs = self.model(audio_input)
            
            # Compute loss based on task type
            if task_type == "emotion_representation":
                # Use mean of last hidden state as emotion representation
                emotion_repr = outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_size]
                # Maximize L2 norm as quality metric
                loss = -emotion_repr.norm(dim=1).mean()  # Negative for maximization
            elif task_type == "hidden_state_variance":
                # Maximize variance across sequence dimension
                hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
                variance = hidden_states.var(dim=1).mean()
                loss = -variance  # Negative for maximization
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            # Backward pass to compute gradients
            loss.backward()
            
        finally:
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()
        
        return activation, gradient
    
    def compute_integrated_gradients(self, 
                                  audio_input: torch.Tensor,
                                  layer_idx: int,
                                  task_type: str = "emotion_representation",
                                  steps: int = 20,
                                  use_abs: bool = True):
        """
        Compute Integrated Gradients for a specific layer
        
        Args:
            audio_input: Preprocessed audio tensor
            layer_idx: Layer index to analyze
            task_type: Type of task for loss function
            steps: Number of steps for path integral
            use_abs: Whether to use absolute values for attributions
            
        Returns:
            Tensor of integrated gradients attributions for each neuron
        """
        # Create baseline (zero tensor)
        baseline = torch.zeros_like(audio_input)
        
        # Initialize integrated gradients
        integrated_grads = None
        
        # Capture original model state
        was_training = self.model.training
        self.model.eval()
        
        try:
            # Compute integrated gradients along the path
            for step in range(1, steps + 1):
                # Create an interpolated input
                alpha = step / steps
                interpolated = baseline + alpha * (audio_input - baseline)
                interpolated.requires_grad_(True)
                
                # Get activation and gradient for this interpolation
                activation, gradient = self.get_layer_activation_gradient(
                    interpolated, layer_idx, task_type)
                
                # Multiply element-wise and aggregate
                if integrated_grads is None:
                    integrated_grads = (activation * gradient) / steps
                else:
                    integrated_grads += (activation * gradient) / steps
                
                # Clean up
                if interpolated.grad is not None:
                    interpolated.grad.zero_()
                
            # Reduce across batch and sequence dimensions to get per-neuron importance
            neuron_importance = integrated_grads.mean(dim=(0, 1))
            
            if use_abs:
                neuron_importance = torch.abs(neuron_importance)
                
        finally:
            # Restore model's original state
            if was_training:
                self.model.train()
            else:
                self.model.eval()
        
        return neuron_importance
    
    def compute_neuron_importance_with_gradient_shap(self,
                                                  audio_input: torch.Tensor,
                                                  layer_idx: int,
                                                  task_type: str = "emotion_representation",
                                                  steps: int = 20) -> Dict[int, Dict[str, float]]:
        """
        Find most important neurons in a specific layer using gradient-based SHAP
        
        Args:
            audio_input: Preprocessed audio tensor
            layer_idx: Layer index to analyze
            task_type: Type of task for loss function
            steps: Number of steps for integrated gradients
            
        Returns:
            Dictionary mapping neuron indices to importance scores
        """
        print(f"\n🔍 Analyzing Layer {layer_idx} with Gradient-SHAP...")
        
        # Compute integrated gradients
        neuron_importance = self.compute_integrated_gradients(
            audio_input=audio_input,
            layer_idx=layer_idx,
            task_type=task_type,
            steps=steps
        )
        
        # Convert to dictionary of importance scores
        importance_dict = {}
        for neuron_idx in range(self.hidden_size):
            score = neuron_importance[neuron_idx].item()
            importance_dict[neuron_idx] = {
                'importance_score': score,
                'layer_idx': layer_idx
            }
        
        # Sort neurons by importance
        sorted_neurons = dict(sorted(importance_dict.items(), 
                                   key=lambda x: x[1]['importance_score'],
                                   reverse=True))
        
        # Print top neurons for this layer
        print(f"  Top 5 neurons in layer {layer_idx}:")
        for i, (neuron_idx, info) in enumerate(list(sorted_neurons.items())[:5]):
            print(f"    #{i+1}: Neuron {neuron_idx} (importance: {info['importance_score']:.6f})")
        
        return sorted_neurons
    
    def analyze_layers_with_gradient_shap(self,
                                       audio_input: torch.Tensor,
                                       layers_to_analyze: List[int] = None,
                                       task_type: str = "emotion_representation",
                                       steps: int = 20) -> Dict:
        """
        Analyze all specified layers using gradient-based SHAP
        
        Args:
            audio_input: Preprocessed audio tensor
            layers_to_analyze: Which layers to analyze (default: all)
            task_type: Type of task for loss function
            steps: Number of steps for integrated gradients
            
        Returns:
            Dictionary with layer-wise importance scores
        """
        if layers_to_analyze is None:
            layers_to_analyze = list(range(self.num_layers))
        
        print(f"\n🚀 Starting Layer-wise Gradient-SHAP Analysis")
        print(f"Analyzing layers: {layers_to_analyze}")
        
        layer_results = {}
        
        for layer_idx in layers_to_analyze:
            layer_importance = self.compute_neuron_importance_with_gradient_shap(
                audio_input=audio_input,
                layer_idx=layer_idx,
                task_type=task_type,
                steps=steps
            )
            layer_results[layer_idx] = layer_importance
        
        return layer_results
    
    def ablate_neurons_in_forward_pass(self,
                                    audio_input: torch.Tensor,
                                    layer_idx: int,
                                    neurons_to_ablate: List[int],
                                    ablation_method: str = "zero"):
        """
        Ablate specified neurons during a forward pass
        
        Args:
            audio_input: Preprocessed audio tensor
            layer_idx: Layer index to analyze
            neurons_to_ablate: List of neuron indices to ablate
            ablation_method: How to ablate neurons ("zero", "mean", or "random")
            
        Returns:
            Model output with ablated neurons
        """
        layer = self.model.encoder.layers[layer_idx]
        
        # Define hook function to ablate neurons
        def ablation_hook(module, input_tensor, output_tensor):
            if isinstance(output_tensor, tuple):
                hidden_states = output_tensor[0].clone()
                other_outputs = output_tensor[1:]
            else:
                hidden_states = output_tensor.clone()
                other_outputs = ()
            
            # Ablate specified neurons
            for neuron_idx in neurons_to_ablate:
                if ablation_method == "zero":
                    hidden_states[:, :, neuron_idx] = 0.0
                elif ablation_method == "mean":
                    mean_val = hidden_states[:, :, neuron_idx].mean()
                    hidden_states[:, :, neuron_idx] = mean_val
                elif ablation_method == "random":
                    std = hidden_states[:, :, neuron_idx].std()
                    mean = hidden_states[:, :, neuron_idx].mean()
                    shape = hidden_states[:, :, neuron_idx].shape
                    random_vals = torch.normal(mean, std, shape, device=hidden_states.device)
                    hidden_states[:, :, neuron_idx] = random_vals
            
            # Return modified output
            if other_outputs:
                return (hidden_states,) + other_outputs
            return hidden_states
        
        # Register hook
        handle = layer.register_forward_hook(ablation_hook)
        
        try:
            # Forward pass with ablation
            with torch.no_grad():
                output = self.model(audio_input)
            return output
        finally:
            # Remove hook
            handle.remove()
    
    def progressive_ablation_of_important_neurons(self,
                                              audio_input: torch.Tensor,
                                              layer_importance: Dict,
                                              task_metric_fn,
                                              top_k_per_layer: int = 10,
                                              ablation_method: str = "zero") -> Dict:
        """
        Progressively ablate important neurons identified by gradient-SHAP
        
        Args:
            audio_input: Preprocessed audio tensor
            layer_importance: Dictionary with layer-wise neuron importance scores
            task_metric_fn: Function to measure task performance
            top_k_per_layer: How many top neurons to ablate per layer
            ablation_method: How to ablate neurons
            
        Returns:
            Dictionary with progressive ablation results
        """
        print(f"\n{'='*60}")
        print("PROGRESSIVE ABLATION OF IMPORTANT NEURONS (GRADIENT-SHAP)")
        print(f"{'='*60}")
        
        # Get baseline performance without ablation
        with torch.no_grad():
            baseline_score = task_metric_fn(self.model, audio_input)
        print(f"Baseline performance: {baseline_score:.6f}")
        
        # Prepare results structure
        results = {
            'baseline_score': baseline_score,
            'layer_analysis': layer_importance,  # Store original importance scores
            'progressive_ablation': {},
            'cumulative_effects': []
        }
        
        active_hooks = []  # Track active ablation hooks
        cumulative_ablated_layers = []
        
        # Process layers in order
        for layer_idx in sorted(layer_importance.keys()):
            print(f"\n📍 Adding Layer {layer_idx} to progressive ablation...")
            
            # Get top neurons from this layer
            layer_neurons = layer_importance[layer_idx]
            top_neurons = list(layer_neurons.keys())[:top_k_per_layer]
            
            # Create progressive ablation hook for this layer
            layer = self.model.encoder.layers[layer_idx]
            
            def create_hook_for_neurons(neurons_list, ablation_method=ablation_method):
                def ablation_hook(module, input_tensor, output_tensor):
                    if isinstance(output_tensor, tuple):
                        hidden_states = output_tensor[0].clone()
                        other_outputs = output_tensor[1:]
                    else:
                        hidden_states = output_tensor.clone()
                        other_outputs = ()
                    
                    # Ablate all specified neurons
                    for neuron_idx in neurons_list:
                        if ablation_method == "zero":
                            hidden_states[:, :, neuron_idx] = 0.0
                        elif ablation_method == "mean":
                            mean_val = hidden_states[:, :, neuron_idx].mean()
                            hidden_states[:, :, neuron_idx] = mean_val
                        elif ablation_method == "random":
                            std = hidden_states[:, :, neuron_idx].std()
                            mean = hidden_states[:, :, neuron_idx].mean()
                            shape = hidden_states[:, :, neuron_idx].shape
                            random_vals = torch.normal(mean, std, shape, device=hidden_states.device)
                            hidden_states[:, :, neuron_idx] = random_vals
                    
                    if other_outputs:
                        return (hidden_states,) + other_outputs
                    return hidden_states
                
                return ablation_hook
            
            # Register hook
            hook = layer.register_forward_hook(create_hook_for_neurons(top_neurons))
            active_hooks.append((layer_idx, hook))
            cumulative_ablated_layers.append(layer_idx)
            
            # Measure cumulative performance with ablation
            with torch.no_grad():
                cumulative_score = task_metric_fn(self.model, audio_input)
            cumulative_drop = baseline_score - cumulative_score  # Positive means performance decreased
            
            progressive_result = {
                'layer_idx': layer_idx,
                'cumulative_layers': cumulative_ablated_layers.copy(),
                'ablated_neurons_this_layer': top_neurons,
                'cumulative_score': cumulative_score,
                'cumulative_drop': cumulative_drop,
                'relative_cumulative_drop': cumulative_drop / baseline_score if baseline_score != 0 else 0,
                'incremental_drop': cumulative_drop - (results['cumulative_effects'][-1]['cumulative_drop'] 
                                                    if results['cumulative_effects'] else 0)
            }
            
            results['progressive_ablation'][layer_idx] = progressive_result
            results['cumulative_effects'].append(progressive_result)
            
            print(f"  Ablated neurons: {top_neurons}")
            print(f"  Cumulative score: {cumulative_score:.6f}")
            print(f"  Cumulative drop: {cumulative_drop:.6f}")
            print(f"  Incremental drop: {progressive_result['incremental_drop']:.6f}")
        
        # Clean up hooks
        print(f"\n🧹 Cleaning up ablation hooks...")
        for layer_idx, hook in active_hooks:
            hook.remove()
        
        return results
    
    def visualize_gradient_shap_and_ablation(self,
                                          layer_importance: Dict,
                                          ablation_results: Dict,
                                          save_path: str = None):
        """
        Visualize gradient-SHAP neuron importance and ablation results
        
        Args:
            layer_importance: Dictionary with layer-wise neuron importance from gradient-SHAP
            ablation_results: Dictionary with progressive ablation results
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Extract data for plotting
        layers = sorted(layer_importance.keys())
        
        # Plot 1: Average importance per layer
        avg_importance_per_layer = []
        max_importance_per_layer = []
        
        for layer_idx in layers:
            layer_data = layer_importance[layer_idx]
            importance_scores = [data['importance_score'] for data in layer_data.values()]
            avg_importance_per_layer.append(np.mean(importance_scores))
            max_importance_per_layer.append(np.max(importance_scores))
        
        axes[0, 0].plot(layers, avg_importance_per_layer, 'b-o', label='Average Importance', linewidth=2)
        axes[0, 0].plot(layers, max_importance_per_layer, 'r-s', label='Max Importance', linewidth=2)
        axes[0, 0].set_title('Gradient-SHAP Neuron Importance Across Layers', fontsize=14)
        axes[0, 0].set_xlabel('Layer Index', fontsize=12)
        axes[0, 0].set_ylabel('Importance Score', fontsize=12)
        axes[0, 0].legend(fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Progressive ablation performance
        if ablation_results['cumulative_effects']:
            cumulative_layers = [len(effect['cumulative_layers']) for effect in ablation_results['cumulative_effects']]
            cumulative_scores = [effect['cumulative_score'] for effect in ablation_results['cumulative_effects']]
            
            axes[0, 1].plot(cumulative_layers, cumulative_scores, 'g-^', linewidth=2, markersize=8)
            axes[0, 1].axhline(y=ablation_results['baseline_score'], color='black', linestyle='--', label='Baseline')
            axes[0, 1].set_title('Progressive Layer Ablation Effect (Gradient-SHAP)', fontsize=14)
            axes[0, 1].set_xlabel('Number of Layers Ablated', fontsize=12)
            axes[0, 1].set_ylabel('Model Performance', fontsize=12)
            axes[0, 1].legend(fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Top neurons heatmap
        layers_to_show = layers[:min(6, len(layers))]
        top_neurons_matrix = []
        
        for layer_idx in layers_to_show:
            layer_data = layer_importance[layer_idx]
            # Get top 10 neurons
            top_10 = list(layer_data.keys())[:10]
            importance_scores = [layer_data[n]['importance_score'] for n in top_10]
            top_neurons_matrix.append(importance_scores)
        
        if top_neurons_matrix:
            im = axes[1, 0].imshow(top_neurons_matrix, cmap='viridis', aspect='auto')
            axes[1, 0].set_title('Top 10 Neurons per Layer (Gradient-SHAP)', fontsize=14)
            axes[1, 0].set_xlabel('Neuron Rank (Top 10)', fontsize=12)
            axes[1, 0].set_ylabel('Layer Index', fontsize=12)
            axes[1, 0].set_yticks(range(len(layers_to_show)))
            axes[1, 0].set_yticklabels(layers_to_show)
            plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Incremental vs Cumulative drops
        if ablation_results['cumulative_effects']:
            incremental_drops = [effect['incremental_drop'] for effect in ablation_results['cumulative_effects']]
            cumulative_drops = [effect['cumulative_drop'] for effect in ablation_results['cumulative_effects']]
            
            x_pos = range(len(incremental_drops))
            
            axes[1, 1].bar([x - 0.2 for x in x_pos], incremental_drops, 0.4, 
                          label='Incremental Drop', alpha=0.7)
            axes[1, 1].plot(x_pos, cumulative_drops, 'r-o', label='Cumulative Drop', linewidth=2)
            axes[1, 1].set_title('Incremental vs Cumulative Performance Drops', fontsize=14)
            axes[1, 1].set_xlabel('Progressive Ablation Step', fontsize=12)
            axes[1, 1].set_ylabel('Performance Drop', fontsize=12)
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels([f"L{layers[i]}" for i in range(len(incremental_drops))])
            axes[1, 1].legend(fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Gradient-SHAP Neuron Importance Analysis and Ablation', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
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


def run_gradient_shap_analysis(ravdess_path: str, 
                             emotion: str = "happy",
                             actor_id: int = None,
                             save_dir: str = "results_gradshap"):
    """
    Run gradient-SHAP neuron importance analysis and progressive ablation
    
    Args:
        ravdess_path: Path to RAVDESS dataset
        emotion: Emotion to analyze
        actor_id: Specific actor ID (optional)
        save_dir: Directory to save results
    """
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("🎵 WAVLM GRADIENT-SHAP NEURON IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Step 1: Initialize analyzer
    print("Step 1: Initializing WavLM analyzer...")
    analyzer = GradientSHAPNeuronAnalysis()
    
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
    
    # Step 4: Create task metric function
    print("Step 4: Creating task metric function...")
    task_metric_fn = analyzer.create_task_metric("emotion_representation")
    
    # Step 5: Run gradient-SHAP analysis
    print("Step 5: Running gradient-SHAP neuron importance analysis...")
    
    # Analyze key layers to save time
    key_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Early, middle, and late layers
    
    layer_importance = analyzer.analyze_layers_with_gradient_shap(
        audio_input=audio_input,
        layers_to_analyze=key_layers,
        task_type="emotion_representation",
        steps=20  # Number of steps for integrated gradients
    )
    
    # Step 6: Run progressive ablation
    print("Step 6: Running progressive ablation of important neurons...")
    ablation_results = analyzer.progressive_ablation_of_important_neurons(
        audio_input=audio_input,
        layer_importance=layer_importance,
        task_metric_fn=task_metric_fn,
        top_k_per_layer=10,  # Ablate top 10 neurons per layer
        ablation_method="zero"
    )
    
    # Step 7: Visualize results
    print("Step 7: Creating visualizations...")
    viz_path = os.path.join(save_dir, f"gradshap_analysis_{emotion}_{metadata['actor_id']}.png")
    analyzer.visualize_gradient_shap_and_ablation(
        layer_importance=layer_importance,
        ablation_results=ablation_results,
        save_path=viz_path
    )
    
    # Step 8: Save results
    print("Step 8: Saving results...")
    results_path = os.path.join(save_dir, f"gradshap_analysis_{emotion}_{metadata['actor_id']}.json")
    analyzer.save_results(ablation_results, metadata, results_path)
    
    # Step 9: Print summary
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Audio: {metadata['emotion_name']} emotion, Actor {metadata['actor_id']} ({metadata['gender']})")
    print(f"Baseline score: {ablation_results['baseline_score']:.6f}")
    print(f"Layers analyzed: {key_layers}")
    
    if ablation_results['cumulative_effects']:
        final_score = ablation_results['cumulative_effects'][-1]['cumulative_score']
        total_drop = ablation_results['cumulative_effects'][-1]['cumulative_drop']
        print(f"Final score after all ablations: {final_score:.6f}")
        print(f"Total performance drop: {total_drop:.6f} ({total_drop/ablation_results['baseline_score']*100:.1f}%)")
    
    return layer_importance, ablation_results, metadata


# Example usage
if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR RAVDESS DATASET LOCATION
    RAVDESS_PATH = r"/path/to/ravdess"  # Update with your RAVDESS path
    
    # Test with different emotions
    emotions_to_test = ["happy"]
    
    for emotion in emotions_to_test:
        print(f"\n{'🎭'*20}")
        print(f"ANALYZING EMOTION: {emotion.upper()}")
        print(f"{'🎭'*20}")
        
        try:
            results = run_gradient_shap_analysis(
                ravdess_path=RAVDESS_PATH,
                emotion=emotion,
                save_dir=f"results_gradshap_{emotion}"
            )
            print(f"✅ Successfully analyzed {emotion} emotion using gradient-SHAP")
            
        except Exception as e:
            print(f"❌ Error analyzing {emotion}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Analysis complete! Check the results_gradshap_* directories for outputs.")