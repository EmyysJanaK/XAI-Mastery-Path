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
import torch.nn.functional as F

class HierarchicalFeatureAnalyzer:
    """
    Hierarchical feature analysis using gradient-based SHAP and sign-flipping ablation
    to study cross-layer effects in the WavLM model on speech data.
    
    This analyzer identifies and categorizes features at different abstraction levels:
    - Low-level: Early layers (0-3) capturing basic acoustic properties
    - Mid-level: Middle layers (4-7) capturing intermediate representations
    - High-level: Later layers (8-11) capturing semantic/emotional concepts
    """
    
    def __init__(self, model_name="microsoft/wavlm-base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize WavLM model with feature extractor"""
        self.device = device
        
        # Initialize feature extractor and model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(device)
        self.model.eval()  # Ensure model is in evaluation mode
        
        # Model architecture info
        self.num_layers = len(self.model.encoder.layers)
        self.hidden_size = self.model.config.hidden_size
        
        # Define hierarchy levels for layers
        self.low_level_layers = list(range(0, 4))  # Layers 0-3
        self.mid_level_layers = list(range(4, 8))  # Layers 4-7
        self.high_level_layers = list(range(8, 12))  # Layers 8-11 (assuming 12 layers)
        
        # Store all activations and frame info across layers
        self.activation_cache = {}
        self.grad_cache = {}
        self.frame_info = {}  # Store frame-specific information
        
        print(f"Hierarchical Feature Analyzer Initialized:")
        print(f"  - Layers: {self.num_layers}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Device: {self.device}")
        print(f"  - Low-level layers: {self.low_level_layers}")
        print(f"  - Mid-level layers: {self.mid_level_layers}")
        print(f"  - High-level layers: {self.high_level_layers}")
    
    def load_audio(self, audio_path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, Dict]:
        """
        Load and preprocess audio file for WavLM
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sampling rate
            
        Returns:
            Tuple of preprocessed audio tensor and metadata
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Get audio metadata
        duration_sec = waveform.shape[1] / sample_rate
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract frames info - WavLM uses 20ms frames with 10ms stride by default
        frame_length_ms = 20
        frame_shift_ms = 10
        
        # Calculate frame information
        frame_length_samples = int(target_sr * frame_length_ms / 1000)
        frame_shift_samples = int(target_sr * frame_shift_ms / 1000)
        num_frames = (waveform.shape[1] - frame_length_samples) // frame_shift_samples + 1
        
        metadata = {
            'file_path': audio_path,
            'original_sample_rate': sample_rate,
            'target_sample_rate': target_sr,
            'duration_sec': duration_sec,
            'frame_info': {
                'frame_length_ms': frame_length_ms,
                'frame_shift_ms': frame_shift_ms,
                'frame_length_samples': frame_length_samples,
                'frame_shift_samples': frame_shift_samples,
                'num_frames': num_frames
            }
        }
        
        # Store frame information
        self.frame_info = metadata['frame_info']
        
        # Use feature extractor to prepare input for the model
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=target_sr, 
            return_tensors="pt"
        )
        
        # Move inputs to device
        audio_input = inputs["input_values"].to(self.device)
        
        return audio_input, metadata
    
    def register_hooks_for_all_layers(self):
        """
        Register hooks to capture activations and gradients for all layers
        
        Returns:
            List of hook handles for cleanup
        """
        # Clear any existing cache
        self.activation_cache = {}
        self.grad_cache = {}
        handles = []
        
        # Register hooks for each layer
        for layer_idx in range(self.num_layers):
            target_layer = self.model.encoder.layers[layer_idx]
            
            # Define forward hook to capture activations
            def make_forward_hook(idx):
                def forward_hook(module, input_tensor, output_tensor):
                    if isinstance(output_tensor, tuple):
                        self.activation_cache[idx] = output_tensor[0].detach()
                    else:
                        self.activation_cache[idx] = output_tensor.detach()
                return forward_hook
            
            # Define backward hook to capture gradients
            def make_backward_hook(idx):
                def backward_hook(module, grad_input, grad_output):
                    if isinstance(grad_output, tuple):
                        self.grad_cache[idx] = grad_output[0].detach()
                    else:
                        self.grad_cache[idx] = grad_output.detach()
                return backward_hook
            
            # Register the hooks
            forward_handle = target_layer.register_forward_hook(make_forward_hook(layer_idx))
            backward_handle = target_layer.register_backward_hook(make_backward_hook(layer_idx))
            
            handles.extend([forward_handle, backward_handle])
            
        return handles
    
    def compute_integrated_gradients_all_layers(self, 
                                             audio_input: torch.Tensor,
                                             task_type: str = "emotion_representation",
                                             steps: int = 20,
                                             use_abs: bool = True) -> Dict[int, torch.Tensor]:
        """
        Compute Integrated Gradients for all layers
        
        Args:
            audio_input: Preprocessed audio tensor
            task_type: Type of task for loss function
            steps: Number of steps for path integral
            use_abs: Whether to use absolute values for attributions
            
        Returns:
            Dictionary mapping layer indices to neuron importance tensors
        """
        # Create baseline (zero tensor)
        baseline = torch.zeros_like(audio_input)
        
        # Initialize storage for integrated gradients for each layer
        integrated_grads = {layer_idx: None for layer_idx in range(self.num_layers)}
        
        # Capture original model state
        was_training = self.model.training
        self.model.eval()
        
        # Register hooks for all layers
        handles = self.register_hooks_for_all_layers()
        
        try:
            # Compute integrated gradients along the path
            for step in range(1, steps + 1):
                # Create an interpolated input
                alpha = step / steps
                interpolated = baseline + alpha * (audio_input - baseline)
                interpolated.requires_grad_(True)
                
                # Forward pass
                outputs = self.model(interpolated)
                
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
                
                # Process activations and gradients for each layer
                for layer_idx in range(self.num_layers):
                    activation = self.activation_cache.get(layer_idx)
                    gradient = self.grad_cache.get(layer_idx)
                    
                    if activation is not None and gradient is not None:
                        # Multiply element-wise and aggregate
                        step_contrib = (activation * gradient) / steps
                        
                        if integrated_grads[layer_idx] is None:
                            integrated_grads[layer_idx] = step_contrib
                        else:
                            integrated_grads[layer_idx] += step_contrib
                
                # Clean up
                if interpolated.grad is not None:
                    interpolated.grad.zero_()
                
            # Reduce across batch and sequence dimensions to get per-neuron importance for each layer
            neuron_importance_by_layer = {}
            for layer_idx, grads in integrated_grads.items():
                if grads is not None:
                    # Average across batch and sequence dimensions
                    neuron_importance = grads.mean(dim=(0, 1))
                    
                    if use_abs:
                        neuron_importance = torch.abs(neuron_importance)
                    
                    neuron_importance_by_layer[layer_idx] = neuron_importance
                
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()
            
            # Restore model's original state
            if was_training:
                self.model.train()
            else:
                self.model.eval()
        
        return neuron_importance_by_layer
    
    def identify_important_neurons_by_level(self, 
                                         neuron_importance: Dict[int, torch.Tensor],
                                         top_k: int = 20) -> Dict[str, Dict[int, List[int]]]:
        """
        Identify the most important neurons at each hierarchical level
        
        Args:
            neuron_importance: Dictionary mapping layer indices to neuron importance tensors
            top_k: Number of top neurons to identify per layer
            
        Returns:
            Dictionary mapping hierarchical levels to important neurons by layer
        """
        important_neurons = {
            'low_level': {},
            'mid_level': {},
            'high_level': {}
        }
        
        # Process each layer
        for layer_idx, importance in neuron_importance.items():
            # Sort neurons by importance
            neuron_indices = torch.argsort(importance, descending=True).tolist()
            top_neurons = neuron_indices[:top_k]
            
            # Assign to appropriate level
            if layer_idx in self.low_level_layers:
                level = 'low_level'
            elif layer_idx in self.mid_level_layers:
                level = 'mid_level'
            elif layer_idx in self.high_level_layers:
                level = 'high_level'
            else:
                continue  # Skip if layer doesn't fit our hierarchy
                
            important_neurons[level][layer_idx] = top_neurons
            
        return important_neurons
    
    def frame_specific_neuron_importance(self, 
                                      audio_input: torch.Tensor,
                                      layer_idx: int) -> torch.Tensor:
        """
        Compute neuron importance for each frame in the input sequence
        
        Args:
            audio_input: Preprocessed audio tensor
            layer_idx: Layer index to analyze
            
        Returns:
            Frame-specific neuron importance tensor [frames, neurons]
        """
        # Prepare model and input
        audio_input = audio_input.clone().detach().requires_grad_(True)
        
        # Get target layer
        target_layer = self.model.encoder.layers[layer_idx]
        
        # Store activations and gradients
        activations = None
        gradients = None
        
        # Define hooks
        def forward_hook(module, input_tensor, output_tensor):
            nonlocal activations
            if isinstance(output_tensor, tuple):
                activations = output_tensor[0].detach()
            else:
                activations = output_tensor.detach()
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            if isinstance(grad_output, tuple):
                gradients = grad_output[0].detach()
            else:
                gradients = grad_output.detach()
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        try:
            # Forward pass
            outputs = self.model(audio_input)
            
            # For each frame, create a loss focused on that frame
            seq_length = outputs.last_hidden_state.shape[1]
            frame_importances = []
            
            # Process each frame individually
            for frame_idx in range(seq_length):
                # Create a mask that focuses only on this frame
                mask = torch.zeros_like(outputs.last_hidden_state)
                mask[:, frame_idx, :] = 1.0
                
                # Compute frame-specific loss
                loss = (outputs.last_hidden_state * mask).sum()
                
                # Backward pass for this frame
                if audio_input.grad is not None:
                    audio_input.grad.zero_()
                loss.backward(retain_graph=(frame_idx < seq_length-1))
                
                # Calculate importance for this frame
                if activations is not None and gradients is not None:
                    # Get only the activations and gradients for this specific frame
                    frame_activation = activations[:, frame_idx, :]
                    frame_gradient = gradients[:, frame_idx, :]
                    
                    # Calculate neuron importance for this frame
                    frame_importance = (frame_activation * frame_gradient).mean(dim=0)
                    frame_importance = torch.abs(frame_importance)
                    
                    frame_importances.append(frame_importance)
            
            # Stack to get frame-specific neuron importance tensor [frames, neurons]
            return torch.stack(frame_importances)
            
        finally:
            forward_handle.remove()
            backward_handle.remove()
    
    def sign_flipping_ablation(self,
                            audio_input: torch.Tensor,
                            layer_idx: int,
                            neurons_to_ablate: List[int],
                            track_lower_layers: bool = True):
        """
        Perform sign-flipping ablation on specified neurons and track effects
        
        Args:
            audio_input: Preprocessed audio tensor
            layer_idx: Layer index to analyze
            neurons_to_ablate: List of neuron indices to ablate
            track_lower_layers: Whether to track effects on lower layers
            
        Returns:
            Dictionary with ablation effects across layers and frames
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Collect baseline activations for all layers (before ablation)
        handles = []
        baseline_activations = {}
        
        # Set up hooks to capture baseline activations
        for l_idx in range(self.num_layers):
            layer = self.model.encoder.layers[l_idx]
            
            def make_capture_hook(idx):
                def hook(module, input_tensor, output_tensor):
                    if isinstance(output_tensor, tuple):
                        baseline_activations[idx] = output_tensor[0].detach().clone()
                    else:
                        baseline_activations[idx] = output_tensor.detach().clone()
                return hook
            
            handle = layer.register_forward_hook(make_capture_hook(l_idx))
            handles.append(handle)
        
        # Get baseline activations
        with torch.no_grad():
            _ = self.model(audio_input)
        
        # Clean up baseline hooks
        for handle in handles:
            handle.remove()
        
        # Now set up ablation hook for target layer and capture hooks for all layers
        ablated_activations = {}
        layers_to_track = range(layer_idx) if track_lower_layers else [layer_idx]
        
        # Define ablation hook for target layer
        def sign_flip_ablation_hook(module, input_tensor, output_tensor):
            if isinstance(output_tensor, tuple):
                hidden_states = output_tensor[0].clone()
                other_outputs = output_tensor[1:]
            else:
                hidden_states = output_tensor.clone()
                other_outputs = ()
            
            # Flip sign of specified neurons (multiply by -1)
            for neuron_idx in neurons_to_ablate:
                hidden_states[:, :, neuron_idx] = -1.0 * hidden_states[:, :, neuron_idx]
            
            # Return modified output
            if other_outputs:
                return (hidden_states,) + other_outputs
            return hidden_states
        
        # Register ablation hook on target layer
        ablation_handle = self.model.encoder.layers[layer_idx].register_forward_hook(sign_flip_ablation_hook)
        
        # Set up capture hooks for layers to track
        capture_handles = []
        for l_idx in layers_to_track:
            layer = self.model.encoder.layers[l_idx]
            
            def make_track_hook(idx):
                def hook(module, input_tensor, output_tensor):
                    if isinstance(output_tensor, tuple):
                        ablated_activations[idx] = output_tensor[0].detach().clone()
                    else:
                        ablated_activations[idx] = output_tensor.detach().clone()
                return hook
            
            handle = layer.register_forward_hook(make_track_hook(l_idx))
            capture_handles.append(handle)
        
        # Get ablated activations
        with torch.no_grad():
            ablated_output = self.model(audio_input)
        
        # Clean up all hooks
        ablation_handle.remove()
        for handle in capture_handles:
            handle.remove()
        
        # Calculate changes in activations
        activation_changes = {}
        for l_idx in layers_to_track:
            if l_idx in baseline_activations and l_idx in ablated_activations:
                # Calculate absolute changes and relative changes
                absolute_change = ablated_activations[l_idx] - baseline_activations[l_idx]
                
                # Calculate relative change
                # Use small epsilon to avoid division by zero
                epsilon = 1e-10
                relative_change = absolute_change / (torch.abs(baseline_activations[l_idx]) + epsilon)
                
                # Average absolute change per frame
                frame_avg_abs_change = absolute_change.abs().mean(dim=2)  # [batch, frames]
                
                # Average relative change per neuron
                neuron_avg_rel_change = relative_change.abs().mean(dim=(0, 1))  # [neurons]
                
                activation_changes[l_idx] = {
                    'absolute_change': absolute_change,
                    'relative_change': relative_change,
                    'frame_avg_abs_change': frame_avg_abs_change,
                    'neuron_avg_rel_change': neuron_avg_rel_change
                }
        
        # Return the output and the activation changes
        return {
            'ablated_output': ablated_output,
            'activation_changes': activation_changes,
            'ablated_layer_idx': layer_idx,
            'ablated_neurons': neurons_to_ablate
        }
    
    def hierarchical_feature_analysis(self, 
                                   audio_input: torch.Tensor, 
                                   audio_metadata: Dict,
                                   top_k: int = 15,
                                   task_type: str = "emotion_representation"):
        """
        Perform hierarchical feature analysis across low, mid, and high-level neurons
        
        Args:
            audio_input: Preprocessed audio tensor
            audio_metadata: Audio metadata including frame information
            top_k: Number of top neurons to analyze per layer
            task_type: Type of task for loss function
            
        Returns:
            Dictionary with hierarchical analysis results
        """
        # Step 1: Compute neuron importance using integrated gradients for all layers
        print("\n🔍 Computing neuron importance using integrated gradients...")
        neuron_importance = self.compute_integrated_gradients_all_layers(
            audio_input=audio_input,
            task_type=task_type,
            steps=20,
            use_abs=True
        )
        
        # Step 2: Identify important neurons at each hierarchical level
        print("\n🔢 Identifying important neurons at each hierarchical level...")
        important_neurons = self.identify_important_neurons_by_level(
            neuron_importance=neuron_importance,
            top_k=top_k
        )
        
        # Step 3: Perform sign-flipping ablation for each level and track effects
        print("\n🧠 Performing sign-flipping ablation and tracking cross-layer effects...")
        
        ablation_results = {
            'low_level': {},
            'mid_level': {},
            'high_level': {}
        }
        
        # For each level, ablate neurons in each layer and track effects
        for level, layers_dict in important_neurons.items():
            print(f"\nAnalyzing {level.replace('_', ' ')} features...")
            
            for layer_idx, neuron_list in layers_dict.items():
                print(f"  Layer {layer_idx}: Ablating {len(neuron_list)} neurons...")
                
                # Perform sign-flipping ablation
                ablation_result = self.sign_flipping_ablation(
                    audio_input=audio_input,
                    layer_idx=layer_idx,
                    neurons_to_ablate=neuron_list,
                    track_lower_layers=True
                )
                
                ablation_results[level][layer_idx] = ablation_result
        
        # Step 4: Analyze frame-specific effects of ablation
        print("\n📊 Analyzing frame-specific effects of ablation...")
        
        frame_effects = {
            'low_level': {},
            'mid_level': {},
            'high_level': {}
        }
        
        # Analyze one representative layer from each level
        representative_layers = {
            'low_level': self.low_level_layers[1],  # e.g., layer 1
            'mid_level': self.mid_level_layers[1],  # e.g., layer 5
            'high_level': self.high_level_layers[1]  # e.g., layer 9
        }
        
        for level, layer_idx in representative_layers.items():
            if layer_idx in important_neurons.get(level, {}):
                neuron_list = important_neurons[level][layer_idx]
                
                print(f"  Analyzing frame effects in {level.replace('_', ' ')} (Layer {layer_idx})...")
                
                # Compute frame-specific neuron importance
                frame_importance = self.frame_specific_neuron_importance(
                    audio_input=audio_input,
                    layer_idx=layer_idx
                )
                
                # Get corresponding ablation results if available
                if layer_idx in ablation_results.get(level, {}):
                    ablation_data = ablation_results[level][layer_idx]
                    
                    # Extract frame-specific changes from ablation
                    frame_changes = None
                    if layer_idx in ablation_data['activation_changes']:
                        frame_changes = ablation_data['activation_changes'][layer_idx]['frame_avg_abs_change']
                    
                    frame_effects[level][layer_idx] = {
                        'frame_importance': frame_importance,
                        'frame_changes': frame_changes,
                        'ablated_neurons': neuron_list
                    }
        
        # Return comprehensive analysis results
        return {
            'neuron_importance': neuron_importance,
            'important_neurons': important_neurons,
            'ablation_results': ablation_results,
            'frame_effects': frame_effects,
            'audio_metadata': audio_metadata
        }
    
    def characterize_hierarchical_features(self, analysis_results: Dict) -> Dict:
        """
        Characterize features at each hierarchical level based on analysis results
        
        Args:
            analysis_results: Results from hierarchical_feature_analysis
            
        Returns:
            Dictionary with feature characterization
        """
        feature_characterization = {
            'low_level': {
                'description': 'Basic acoustic properties (e.g., frequency, amplitude)',
                'ablation_impact': {
                    'self_layer': 0.0,
                    'lower_layers': 0.0,
                    'higher_layers': 0.0
                },
                'frame_sensitivity': 0.0,
                'key_neurons': []
            },
            'mid_level': {
                'description': 'Intermediate speech properties (e.g., phonemes, prosody)',
                'ablation_impact': {
                    'self_layer': 0.0,
                    'lower_layers': 0.0,
                    'higher_layers': 0.0
                },
                'frame_sensitivity': 0.0,
                'key_neurons': []
            },
            'high_level': {
                'description': 'Semantic/emotional concepts (e.g., emotion, speaker traits)',
                'ablation_impact': {
                    'self_layer': 0.0,
                    'lower_layers': 0.0,
                    'higher_layers': 0.0
                },
                'frame_sensitivity': 0.0,
                'key_neurons': []
            }
        }
        
        # Extract ablation impacts for each level
        for level in ['low_level', 'mid_level', 'high_level']:
            # Skip if no ablation results for this level
            if level not in analysis_results['ablation_results']:
                continue
                
            layer_results = analysis_results['ablation_results'][level]
            
            self_impacts = []
            lower_impacts = []
            higher_impacts = []
            
            for layer_idx, ablation_data in layer_results.items():
                activation_changes = ablation_data['activation_changes']
                
                # Measure impact on self layer
                if layer_idx in activation_changes:
                    self_impact = activation_changes[layer_idx]['neuron_avg_rel_change'].mean().item()
                    self_impacts.append(self_impact)
                
                # Measure impact on lower layers
                lower_layer_impacts = []
                for l_idx in range(layer_idx):
                    if l_idx in activation_changes:
                        impact = activation_changes[l_idx]['neuron_avg_rel_change'].mean().item()
                        lower_layer_impacts.append(impact)
                
                if lower_layer_impacts:
                    lower_impacts.append(np.mean(lower_layer_impacts))
                    
                # Note: We can't directly measure impact on higher layers through our ablation
                # as the forward pass only affects subsequent computations
            
            # Average impacts across layers in this level
            if self_impacts:
                feature_characterization[level]['ablation_impact']['self_layer'] = np.mean(self_impacts)
            if lower_impacts:
                feature_characterization[level]['ablation_impact']['lower_layers'] = np.mean(lower_impacts)
            
            # Frame sensitivity
            if level in analysis_results['frame_effects']:
                frame_effects = analysis_results['frame_effects'][level]
                
                frame_sensitivities = []
                for layer_idx, effects in frame_effects.items():
                    if 'frame_importance' in effects:
                        # Measure variation across frames (higher variation = higher frame sensitivity)
                        frame_variation = effects['frame_importance'].std(dim=0).mean().item()
                        frame_sensitivities.append(frame_variation)
                
                if frame_sensitivities:
                    feature_characterization[level]['frame_sensitivity'] = np.mean(frame_sensitivities)
            
            # Extract key neurons (those that appear most frequently across layers at this level)
            all_important_neurons = []
            for layer_idx, neurons in analysis_results['important_neurons'].get(level, {}).items():
                all_important_neurons.extend([(layer_idx, n) for n in neurons])
            
            # Take the top 10 most important neurons
            feature_characterization[level]['key_neurons'] = all_important_neurons[:10]
        
        return feature_characterization
    
    def visualize_hierarchical_analysis(self, 
                                      analysis_results: Dict, 
                                      feature_characterization: Dict,
                                      save_path: str = None):
        """
        Visualize results of hierarchical feature analysis
        
        Args:
            analysis_results: Results from hierarchical_feature_analysis
            feature_characterization: Results from characterize_hierarchical_features
            save_path: Path to save visualization
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Define grid layout
        gs = plt.GridSpec(3, 3, figure=fig)
        
        # Plot 1: Neuron importance by hierarchical level
        ax1 = fig.add_subplot(gs[0, 0])
        
        levels = ['low_level', 'mid_level', 'high_level']
        level_importances = []
        
        for level in levels:
            level_importance = 0.0
            layer_count = 0
            
            if level == 'low_level':
                layer_range = self.low_level_layers
            elif level == 'mid_level':
                layer_range = self.mid_level_layers
            else:  # high_level
                layer_range = self.high_level_layers
            
            for layer_idx in layer_range:
                if layer_idx in analysis_results['neuron_importance']:
                    layer_importance = analysis_results['neuron_importance'][layer_idx].mean().item()
                    level_importance += layer_importance
                    layer_count += 1
            
            if layer_count > 0:
                level_importances.append(level_importance / layer_count)
            else:
                level_importances.append(0.0)
        
        # Plot bar chart of level importances
        x_pos = np.arange(len(levels))
        ax1.bar(x_pos, level_importances, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['Low-level', 'Mid-level', 'High-level'])
        ax1.set_title('Average Neuron Importance by Hierarchical Level', fontsize=14)
        ax1.set_ylabel('Average Importance', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Ablation impact across hierarchical levels
        ax2 = fig.add_subplot(gs[0, 1:])
        
        # Extract ablation impact data
        level_names = ['Low-level', 'Mid-level', 'High-level']
        self_impacts = []
        lower_impacts = []
        
        for level in levels:
            self_impacts.append(feature_characterization[level]['ablation_impact']['self_layer'])
            lower_impacts.append(feature_characterization[level]['ablation_impact']['lower_layers'])
        
        x = np.arange(len(level_names))
        width = 0.35
        
        ax2.bar(x - width/2, self_impacts, width, label='Impact on Self Layer', color='#1f77b4', alpha=0.7)
        ax2.bar(x + width/2, lower_impacts, width, label='Impact on Lower Layers', color='#ff7f0e', alpha=0.7)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(level_names)
        ax2.set_title('Sign-Flipping Ablation Impact by Hierarchical Level', fontsize=14)
        ax2.set_ylabel('Average Relative Change', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Frame sensitivity across hierarchical levels
        ax3 = fig.add_subplot(gs[1, 0])
        
        frame_sensitivities = [feature_characterization[level]['frame_sensitivity'] for level in levels]
        
        ax3.bar(x_pos, frame_sensitivities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(['Low-level', 'Mid-level', 'High-level'])
        ax3.set_title('Frame Sensitivity by Hierarchical Level', fontsize=14)
        ax3.set_ylabel('Frame Variation', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Neuron importance distribution across layers
        ax4 = fig.add_subplot(gs[1, 1:])
        
        # Extract layer-wise neuron importance data
        layer_indices = sorted(analysis_results['neuron_importance'].keys())
        layer_avg_importances = [analysis_results['neuron_importance'][idx].mean().item() for idx in layer_indices]
        layer_max_importances = [analysis_results['neuron_importance'][idx].max().item() for idx in layer_indices]
        
        ax4.plot(layer_indices, layer_avg_importances, 'o-', label='Average Importance', color='#1f77b4')
        ax4.plot(layer_indices, layer_max_importances, 's-', label='Max Importance', color='#ff7f0e')
        
        # Add vertical separators for hierarchical levels
        ax4.axvspan(-0.5, 3.5, alpha=0.2, color='blue', label='Low-level')
        ax4.axvspan(3.5, 7.5, alpha=0.2, color='orange', label='Mid-level')
        ax4.axvspan(7.5, 11.5, alpha=0.2, color='green', label='High-level')
        
        ax4.set_title('Neuron Importance Distribution Across Layers', fontsize=14)
        ax4.set_xlabel('Layer Index', fontsize=12)
        ax4.set_ylabel('Importance Score', fontsize=12)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Frame-specific effects for a representative layer from each level
        ax5 = fig.add_subplot(gs[2, :])
        
        # Check if we have frame effect data
        if any(analysis_results['frame_effects']):
            # Collect frame effect data
            frame_data = []
            labels = []
            
            for level in levels:
                if level in analysis_results['frame_effects'] and analysis_results['frame_effects'][level]:
                    # Just take the first layer in this level
                    layer_idx, effects = next(iter(analysis_results['frame_effects'][level].items()))
                    
                    if 'frame_importance' in effects:
                        # Get frame-wise mean importance
                        frame_wise_importance = effects['frame_importance'].mean(dim=1)
                        frame_data.append(frame_wise_importance.cpu().numpy())
                        labels.append(f"{level.replace('_', ' ').title()} (Layer {layer_idx})")
            
            if frame_data:
                # Plot frame-specific effects
                for i, (data, label) in enumerate(zip(frame_data, labels)):
                    frames = np.arange(len(data))
                    ax5.plot(frames, data, '-', label=label, linewidth=2)
                
                ax5.set_title('Frame-Specific Neuron Importance', fontsize=14)
                ax5.set_xlabel('Frame Index', fontsize=12)
                ax5.set_ylabel('Average Importance', fontsize=12)
                ax5.legend(fontsize=11)
                ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No frame data available", ha='center', va='center', fontsize=14)
            ax5.axis('off')
        
        plt.suptitle('Hierarchical Feature Analysis with Sign-Flipping Ablation', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def interpret_hierarchical_features(self, feature_characterization: Dict) -> Dict:
        """
        Provide interpretation of the hierarchical features based on analysis
        
        Args:
            feature_characterization: Results from characterize_hierarchical_features
            
        Returns:
            Dictionary with interpretations for each level
        """
        interpretations = {}
        
        for level, data in feature_characterization.items():
            level_name = level.replace('_', ' ').title()
            
            # Interpret self-impact vs. lower-impact
            self_impact = data['ablation_impact']['self_layer']
            lower_impact = data['ablation_impact']['lower_layers']
            frame_sensitivity = data['frame_sensitivity']
            
            # Determine feature stability across frames
            if frame_sensitivity > 0.3:
                frame_stability = "highly frame-dependent"
            elif frame_sensitivity > 0.15:
                frame_stability = "moderately frame-dependent"
            else:
                frame_stability = "relatively frame-invariant"
            
            # Determine propagation behavior
            if lower_impact > self_impact * 0.8:
                propagation = "strongly propagates to lower layers"
            elif lower_impact > self_impact * 0.4:
                propagation = "moderately propagates to lower layers"
            else:
                propagation = "has limited effect on lower layers"
            
            # Create interpretation
            interpretation = f"{level_name} features are {frame_stability} and {propagation}. "
            
            if level == 'low_level':
                interpretation += ("These likely represent basic acoustic properties like frequency bands and "
                                 "amplitude patterns that form the foundation of audio processing.")
            elif level == 'mid_level':
                interpretation += ("These likely represent intermediate speech patterns such as phonetic units, "
                                 "prosodic elements, and voice characteristics.")
            else:  # high_level
                interpretation += ("These likely represent semantic or emotional concepts that integrate "
                                 "information across multiple frames to capture high-level speech attributes.")
            
            interpretations[level] = interpretation
        
        return interpretations
    
    def save_analysis_results(self, 
                           analysis_results: Dict, 
                           feature_characterization: Dict,
                           interpretations: Dict, 
                           save_path: str):
        """
        Save analysis results to JSON file
        
        Args:
            analysis_results: Results from hierarchical_feature_analysis
            feature_characterization: Results from characterize_hierarchical_features
            interpretations: Results from interpret_hierarchical_features
            save_path: Path to save results
        """
        # Convert numpy/torch types to Python native types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
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
        
        # Create a save-friendly version of results
        save_data = {
            'audio_metadata': analysis_results['audio_metadata'],
            'important_neurons': convert_for_json(analysis_results['important_neurons']),
            'feature_characterization': convert_for_json(feature_characterization),
            'interpretations': interpretations,
            'model_info': {
                'num_layers': self.num_layers,
                'hidden_size': self.hidden_size,
                'device': str(self.device)
            }
        }
        
        # Also save some summarized ablation results if available
        if 'ablation_results' in analysis_results:
            ablation_summary = {}
            
            for level, layers_dict in analysis_results['ablation_results'].items():
                ablation_summary[level] = {}
                
                for layer_idx, ablation_data in layers_dict.items():
                    # Include key summary metrics
                    layer_summary = {
                        'ablated_neurons': ablation_data['ablated_neurons'],
                        'layer_impacts': {}
                    }
                    
                    # Summarize impacts on different layers
                    for impact_layer, change_data in ablation_data['activation_changes'].items():
                        layer_summary['layer_impacts'][impact_layer] = {
                            'avg_abs_change': float(change_data['neuron_avg_rel_change'].mean().item()),
                            'max_abs_change': float(change_data['neuron_avg_rel_change'].max().item())
                        }
                    
                    ablation_summary[level][layer_idx] = layer_summary
            
            save_data['ablation_summary'] = ablation_summary
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Analysis results saved to: {save_path}")


def run_hierarchical_feature_analysis(audio_path: str, save_dir: str = "results_hierarchical"):
    """
    Run hierarchical feature analysis on an audio file
    
    Args:
        audio_path: Path to audio file
        save_dir: Directory to save results
    """
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("🔍 HIERARCHICAL FEATURE ANALYSIS WITH SIGN-FLIPPING ABLATION")
    print("="*70)
    
    # Step 1: Initialize analyzer
    print("\nStep 1: Initializing Hierarchical Feature Analyzer...")
    analyzer = HierarchicalFeatureAnalyzer()
    
    # Step 2: Load and preprocess audio
    print("\nStep 2: Loading and preprocessing audio...")
    audio_input, audio_metadata = analyzer.load_audio(audio_path)
    print(f"Loaded audio: {audio_path}")
    print(f"Audio duration: {audio_metadata['duration_sec']:.2f} seconds")
    print(f"Estimated frames: {audio_metadata['frame_info']['num_frames']}")
    print(f"Audio tensor shape: {audio_input.shape}")
    
    # Step 3: Perform hierarchical feature analysis
    print("\nStep 3: Performing hierarchical feature analysis...")
    analysis_results = analyzer.hierarchical_feature_analysis(
        audio_input=audio_input,
        audio_metadata=audio_metadata,
        top_k=15  # Analyze top 15 neurons per layer
    )
    
    # Step 4: Characterize features at each hierarchical level
    print("\nStep 4: Characterizing features at each hierarchical level...")
    feature_characterization = analyzer.characterize_hierarchical_features(analysis_results)
    
    # Step 5: Interpret hierarchical features
    print("\nStep 5: Interpreting hierarchical features...")
    interpretations = analyzer.interpret_hierarchical_features(feature_characterization)
    
    # Print interpretations
    for level, interpretation in interpretations.items():
        level_name = level.replace('_', ' ').title()
        print(f"\n{level_name} Features:")
        print(f"  {interpretation}")
    
    # Step 6: Visualize results
    print("\nStep 6: Visualizing analysis results...")
    viz_path = os.path.join(save_dir, f"hierarchical_analysis_{Path(audio_path).stem}.png")
    analyzer.visualize_hierarchical_analysis(
        analysis_results=analysis_results,
        feature_characterization=feature_characterization,
        save_path=viz_path
    )
    
    # Step 7: Save results
    print("\nStep 7: Saving analysis results...")
    results_path = os.path.join(save_dir, f"hierarchical_analysis_{Path(audio_path).stem}.json")
    analyzer.save_analysis_results(
        analysis_results=analysis_results,
        feature_characterization=feature_characterization,
        interpretations=interpretations,
        save_path=results_path
    )
    
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Audio file: {Path(audio_path).name}")
    print(f"Analysis approach: Gradient-SHAP + Sign-flipping ablation")
    print(f"Results saved to: {save_dir}")
    
    # Show key insights from the analysis
    print("\nKey Insights:")
    
    # Compare impact of sign-flipping across levels
    low_impact = feature_characterization['low_level']['ablation_impact']['lower_layers']
    mid_impact = feature_characterization['mid_level']['ablation_impact']['lower_layers']
    high_impact = feature_characterization['high_level']['ablation_impact']['lower_layers']
    
    print(f"- Low-level features impact on lower layers: {low_impact:.4f}")
    print(f"- Mid-level features impact on lower layers: {mid_impact:.4f}")
    print(f"- High-level features impact on lower layers: {high_impact:.4f}")
    
    # Compare frame sensitivity
    low_frame = feature_characterization['low_level']['frame_sensitivity']
    mid_frame = feature_characterization['mid_level']['frame_sensitivity']
    high_frame = feature_characterization['high_level']['frame_sensitivity']
    
    print(f"- Low-level frame sensitivity: {low_frame:.4f}")
    print(f"- Mid-level frame sensitivity: {mid_frame:.4f}")
    print(f"- High-level frame sensitivity: {high_frame:.4f}")
    
    return analysis_results, feature_characterization, interpretations


if __name__ == "__main__":
    # Example usage - replace with your RAVDESS audio path
    AUDIO_PATH = r"/path/to/ravdess/Actor_01/03-01-01-01-01-01-01.wav"
    
    try:
        results = run_hierarchical_feature_analysis(
            audio_path=AUDIO_PATH,
            save_dir="results_hierarchical"
        )
        print("\n✅ Successfully completed hierarchical feature analysis")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()