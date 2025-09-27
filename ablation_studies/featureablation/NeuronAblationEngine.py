#!/usr/bin/env python3
# filepath: c:\Users\janak\Desktop\GitHub\New folder\XAI-Mastery-Path\ablation_studies\NeuronAblation.py

"""
Neuron Ablation: Remove specific neurons within layers and measure impact on model performance
This is different from Feature Ablation which removes input features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
from copy import deepcopy
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class NeuronAblationEngine:
    """
    Core engine for performing neuron ablation on neural networks
    """
    
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the neuron ablation engine
        
        Args:
            model: PyTorch model to analyze
            device: Device to run computations on
        """
        self.original_model = model.to(device)
        self.device = device
        self.model_architecture = self._analyze_architecture()
        self.ablation_hooks = []  # Store registered hooks
        
    def _analyze_architecture(self):
        """Analyze model architecture to identify ablatable components"""
        architecture = {
            'total_parameters': sum(p.numel() for p in self.original_model.parameters()),
            'layers': [],
            'ablatable_components': {}
        }
        
        # For transformer models like WavLM
        if hasattr(self.original_model, 'encoder') and hasattr(self.original_model.encoder, 'layers'):
            for i, layer in enumerate(self.original_model.encoder.layers):
                layer_info = {
                    'layer_idx': i,
                    'type': 'transformer_layer',
                    'hidden_size': getattr(self.original_model.config, 'hidden_size', None),
                    'intermediate_size': getattr(self.original_model.config, 'intermediate_size', None),
                    'num_attention_heads': getattr(self.original_model.config, 'num_attention_heads', None)
                }
                architecture['layers'].append(layer_info)
        
        return architecture
    
    def get_ablatable_neurons(self, layer_idx: int) -> Dict[str, Any]:
        """
        Get information about neurons that can be ablated in a specific layer
        
        Args:
            layer_idx: Index of the layer to analyze
            
        Returns:
            Dictionary with ablatable neuron information
        """
        if not hasattr(self.original_model.encoder, 'layers'):
            raise ValueError("Model architecture not supported for neuron ablation")
        
        if layer_idx >= len(self.original_model.encoder.layers):
            raise ValueError(f"Layer {layer_idx} does not exist")
        
        layer = self.original_model.encoder.layers[layer_idx]
        
        ablatable_info = {
            'layer_idx': layer_idx,
            'hidden_neurons': {
                'count': self.original_model.config.hidden_size,
                'description': 'Main hidden state neurons (output of each transformer layer)'
            },
            'attention_neurons': {
                'count': self.original_model.config.num_attention_heads * 
                        (self.original_model.config.hidden_size // self.original_model.config.num_attention_heads),
                'num_heads': self.original_model.config.num_attention_heads,
                'head_size': self.original_model.config.hidden_size // self.original_model.config.num_attention_heads,
                'description': 'Attention mechanism neurons'
            },
            'ffn_neurons': {
                'count': self.original_model.config.intermediate_size,
                'description': 'Feed-forward network intermediate neurons'
            }
        }
        
        return ablatable_info
    
    def ablate_hidden_neurons(self, 
                            layer_idx: int, 
                            neuron_indices: Union[int, List[int]], 
                            ablation_method: str = "zero") -> None:
        """
        Ablate specific hidden state neurons in a transformer layer
        
        Args:
            layer_idx: Which layer to ablate
            neuron_indices: Which neurons to ablate (hidden dimensions)
            ablation_method: How to ablate ('zero', 'mean', 'random', 'noise')
        """
        if isinstance(neuron_indices, int):
            neuron_indices = [neuron_indices]
        
        layer = self.original_model.encoder.layers[layer_idx]
        
        def hidden_ablation_hook(module, input_tensor, output_tensor):
            """Hook to ablate hidden state neurons"""
            # Handle transformer layer output format
            if isinstance(output_tensor, tuple):
                hidden_states = output_tensor[0].clone()  # [batch, seq_len, hidden_size]
                other_outputs = output_tensor[1:]
            else:
                hidden_states = output_tensor.clone()
                other_outputs = ()
            
            # Apply ablation to specified neurons
            for neuron_idx in neuron_indices:
                if neuron_idx < hidden_states.shape[-1]:
                    if ablation_method == "zero":
                        hidden_states[:, :, neuron_idx] = 0.0
                    elif ablation_method == "mean":
                        # Replace with mean activation across batch and sequence
                        mean_val = hidden_states[:, :, neuron_idx].mean()
                        hidden_states[:, :, neuron_idx] = mean_val
                    elif ablation_method == "random":
                        # Replace with random values from normal distribution
                        std = hidden_states[:, :, neuron_idx].std()
                        mean = hidden_states[:, :, neuron_idx].mean()
                        shape = hidden_states[:, :, neuron_idx].shape
                        random_vals = torch.normal(mean, std, shape, device=hidden_states.device)
                        hidden_states[:, :, neuron_idx] = random_vals
                    elif ablation_method == "noise":
                        # Add noise to existing values
                        noise_std = hidden_states[:, :, neuron_idx].std() * 0.1
                        noise = torch.normal(0, noise_std, hidden_states[:, :, neuron_idx].shape, 
                                           device=hidden_states.device)
                        hidden_states[:, :, neuron_idx] += noise
            
            # Return modified output in correct format
            if other_outputs:
                return (hidden_states,) + other_outputs
            return hidden_states
        
        # Register hook
        handle = layer.register_forward_hook(hidden_ablation_hook)
        self.ablation_hooks.append(handle)
    
    def ablate_attention_heads(self, 
                             layer_idx: int, 
                             head_indices: Union[int, List[int]], 
                             ablation_method: str = "zero") -> None:
        """
        Ablate specific attention heads
        
        Args:
            layer_idx: Which layer to ablate
            head_indices: Which attention heads to ablate
            ablation_method: How to ablate the attention heads
        """
        if isinstance(head_indices, int):
            head_indices = [head_indices]
        
        layer = self.original_model.encoder.layers[layer_idx]
        head_size = self.original_model.config.hidden_size // self.original_model.config.num_attention_heads
        
        def attention_ablation_hook(module, input_tensor, output_tensor):
            """Hook to ablate attention head outputs"""
            if isinstance(output_tensor, tuple):
                attention_output = output_tensor[0].clone()  # [batch, seq_len, hidden_size]
                other_outputs = output_tensor[1:]
            else:
                attention_output = output_tensor.clone()
                other_outputs = ()
            
            # Ablate specified attention heads
            for head_idx in head_indices:
                if head_idx < self.original_model.config.num_attention_heads:
                    start_idx = head_idx * head_size
                    end_idx = (head_idx + 1) * head_size
                    
                    if ablation_method == "zero":
                        attention_output[:, :, start_idx:end_idx] = 0.0
                    elif ablation_method == "mean":
                        mean_val = attention_output[:, :, start_idx:end_idx].mean()
                        attention_output[:, :, start_idx:end_idx] = mean_val
                    elif ablation_method == "random":
                        shape = attention_output[:, :, start_idx:end_idx].shape
                        std = attention_output[:, :, start_idx:end_idx].std()
                        mean = attention_output[:, :, start_idx:end_idx].mean()
                        random_vals = torch.normal(mean, std, shape, device=attention_output.device)
                        attention_output[:, :, start_idx:end_idx] = random_vals
            
            if other_outputs:
                return (attention_output,) + other_outputs
            return attention_output
        
        # Register hook on attention layer
        if hasattr(layer, 'attention'):
            handle = layer.attention.register_forward_hook(attention_ablation_hook)
            self.ablation_hooks.append(handle)
    
    def ablate_ffn_neurons(self, 
                          layer_idx: int, 
                          neuron_indices: Union[int, List[int]], 
                          ablation_method: str = "zero") -> None:
        """
        Ablate feed-forward network neurons
        
        Args:
            layer_idx: Which layer to ablate
            neuron_indices: Which FFN neurons to ablate
            ablation_method: How to ablate the neurons
        """
        if isinstance(neuron_indices, int):
            neuron_indices = [neuron_indices]
        
        layer = self.original_model.encoder.layers[layer_idx]
        
        def ffn_ablation_hook(module, input_tensor, output_tensor):
            """Hook to ablate FFN intermediate neurons"""
            # This targets the intermediate layer in the FFN
            if hasattr(module, 'weight') and len(output_tensor.shape) == 3:
                # output_tensor: [batch, seq_len, intermediate_size]
                modified_output = output_tensor.clone()
                
                for neuron_idx in neuron_indices:
                    if neuron_idx < modified_output.shape[-1]:
                        if ablation_method == "zero":
                            modified_output[:, :, neuron_idx] = 0.0
                        elif ablation_method == "mean":
                            mean_val = modified_output[:, :, neuron_idx].mean()
                            modified_output[:, :, neuron_idx] = mean_val
                        elif ablation_method == "random":
                            shape = modified_output[:, :, neuron_idx].shape
                            std = modified_output[:, :, neuron_idx].std()
                            mean = modified_output[:, :, neuron_idx].mean()
                            random_vals = torch.normal(mean, std, shape, device=modified_output.device)
                            modified_output[:, :, neuron_idx] = random_vals
                
                return modified_output
            return output_tensor
        
        # Register hook on FFN intermediate layer
        if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'intermediate_dense'):
            handle = layer.feed_forward.intermediate_dense.register_forward_hook(ffn_ablation_hook)
            self.ablation_hooks.append(handle)
    
    def clear_ablations(self):
        """Remove all ablation hooks"""
        for handle in self.ablation_hooks:
            handle.remove()
        self.ablation_hooks.clear()
    
    def analyze_neuron_importance(self, 
                                 input_data, 
                                 layer_idx: int,
                                 neuron_type: str = "hidden",  # "hidden", "attention", "ffn"
                                 task_metric_fn: Callable = None,
                                 num_neurons_to_test: int = 50,
                                 ablation_method: str = "zero") -> Dict[int, Dict[str, float]]:
        """
        Systematically ablate individual neurons to measure their importance
        
        Args:
            input_data: Input tensor for the model
            layer_idx: Which layer to analyze
            neuron_type: Type of neurons to analyze
            task_metric_fn: Function to measure task performance
            num_neurons_to_test: Number of neurons to test
            ablation_method: Method for ablation
            
        Returns:
            Dictionary mapping neuron indices to importance scores
        """
        if task_metric_fn is None:
            # Default metric: use model output norm
            def default_metric(model, inputs):
                with torch.no_grad():
                    outputs = model(inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        return outputs.last_hidden_state.norm().item()
                    return outputs.norm().item() if torch.is_tensor(outputs) else 0.0
            task_metric_fn = default_metric
        
        # Get baseline performance
        baseline_score = task_metric_fn(self.original_model, input_data)
        
        neuron_importance = {}
        ablatable_info = self.get_ablatable_neurons(layer_idx)
        
        # Determine which neurons to test based on type
        if neuron_type == "hidden":
            max_neurons = ablatable_info['hidden_neurons']['count']
        elif neuron_type == "attention":
            max_neurons = ablatable_info['attention_neurons']['num_heads']
        elif neuron_type == "ffn":
            max_neurons = ablatable_info['ffn_neurons']['count']
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
        
        num_to_test = min(num_neurons_to_test, max_neurons)
        
        print(f"Analyzing {num_to_test} {neuron_type} neurons in layer {layer_idx}...")
        print(f"Baseline performance: {baseline_score:.6f}")
        
        for neuron_idx in range(num_to_test):
            if neuron_idx % 10 == 0:
                print(f"  Testing neuron {neuron_idx}/{num_to_test}")
            
            # Clear previous ablations
            self.clear_ablations()
            
            # Apply ablation to current neuron
            if neuron_type == "hidden":
                self.ablate_hidden_neurons(layer_idx, neuron_idx, ablation_method)
            elif neuron_type == "attention":
                self.ablate_attention_heads(layer_idx, neuron_idx, ablation_method)
            elif neuron_type == "ffn":
                self.ablate_ffn_neurons(layer_idx, neuron_idx, ablation_method)
            
            # Measure performance with ablated neuron
            try:
                ablated_score = task_metric_fn(self.original_model, input_data)
                importance = baseline_score - ablated_score
                
                neuron_importance[neuron_idx] = {
                    'importance_score': importance,
                    'baseline_score': baseline_score,
                    'ablated_score': ablated_score,
                    'relative_drop': (importance / baseline_score) if baseline_score != 0 else 0.0,
                    'neuron_type': neuron_type
                }
            except Exception as e:
                print(f"Error testing neuron {neuron_idx}: {e}")
                neuron_importance[neuron_idx] = {
                    'importance_score': 0.0,
                    'baseline_score': baseline_score,
                    'ablated_score': baseline_score,
                    'relative_drop': 0.0,
                    'neuron_type': neuron_type,
                    'error': str(e)
                }
            
            # Clear ablation for next iteration
            self.clear_ablations()
        
        return neuron_importance
    
    def compare_ablation_methods(self, 
                               input_data,
                               layer_idx: int, 
                               neuron_idx: int,
                               neuron_type: str = "hidden",
                               task_metric_fn: Callable = None) -> Dict[str, Dict[str, float]]:
        """
        Compare different ablation methods on the same neuron
        """
        if task_metric_fn is None:
            def default_metric(model, inputs):
                with torch.no_grad():
                    outputs = model(inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        return outputs.last_hidden_state.norm().item()
                    return outputs.norm().item()
            task_metric_fn = default_metric
        
        baseline_score = task_metric_fn(self.original_model, input_data)
        methods = ["zero", "mean", "random", "noise"]
        results = {}
        
        for method in methods:
            self.clear_ablations()
            
            # Apply ablation with current method
            if neuron_type == "hidden":
                self.ablate_hidden_neurons(layer_idx, neuron_idx, method)
            elif neuron_type == "attention":
                self.ablate_attention_heads(layer_idx, neuron_idx, method)
            elif neuron_type == "ffn":
                self.ablate_ffn_neurons(layer_idx, neuron_idx, method)
            
            try:
                ablated_score = task_metric_fn(self.original_model, input_data)
                importance = baseline_score - ablated_score
                
                results[method] = {
                    'score': ablated_score,
                    'importance': importance,
                    'relative_drop': (importance / baseline_score) if baseline_score != 0 else 0.0
                }
            except Exception as e:
                results[method] = {
                    'score': baseline_score,
                    'importance': 0.0,
                    'relative_drop': 0.0,
                    'error': str(e)
                }
        
        self.clear_ablations()
        return results


class WavLMNeuronAblation:
    """
    Specialized neuron ablation for WavLM models with RAVDESS dataset
    """
    
    def __init__(self, model_name="microsoft/wavlm-base"):
        """Initialize WavLM model and ablation engine"""
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name)
        self.ablation_engine = NeuronAblationEngine(self.model)
        
        print(f"WavLM Model loaded:")
        print(f"  - Layers: {len(self.model.encoder.layers)}")
        print(f"  - Hidden size: {self.model.config.hidden_size}")
        print(f"  - Attention heads: {self.model.config.num_attention_heads}")
        print(f"  - FFN size: {self.model.config.intermediate_size}")
    
    def preprocess_audio(self, audio_input, sample_rate=16000):
        """Preprocess audio for WavLM"""
        if isinstance(audio_input, str):
            import torchaudio
            waveform, sr = torchaudio.load(audio_input)
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            audio_array = waveform.squeeze().numpy()
        else:
            audio_array = np.array(audio_input).squeeze()
        
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        return inputs["input_values"]
    
    def create_emotion_task_metric(self):
        """Create a simple emotion classification metric"""
        def emotion_metric(model, audio_input):
            with torch.no_grad():
                outputs = model(audio_input)
                # Use the mean of last hidden state as emotion representation
                emotion_repr = outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_size]
                # Simple metric: L2 norm of emotion representation
                return emotion_repr.norm(dim=1).mean().item()
        return emotion_metric
    
    def analyze_emotion_processing_neurons(self, 
                                         audio_input,
                                         layer_idx: int = 6,
                                         num_neurons: int = 50) -> Dict[str, Any]:
        """
        Analyze which neurons are important for emotion processing
        """
        processed_audio = self.preprocess_audio(audio_input)
        emotion_metric = self.create_emotion_task_metric()
        
        print(f"\n=== ANALYZING EMOTION PROCESSING NEURONS ===")
        print(f"Layer: {layer_idx}, Audio shape: {processed_audio.shape}")
        
        results = {}
        
        # Analyze different neuron types
        for neuron_type in ["hidden", "attention"]:  # Skip FFN for now due to complexity
            print(f"\n--- Analyzing {neuron_type} neurons ---")
            
            importance_scores = self.ablation_engine.analyze_neuron_importance(
                input_data=processed_audio,
                layer_idx=layer_idx,
                neuron_type=neuron_type,
                task_metric_fn=emotion_metric,
                num_neurons_to_test=num_neurons,
                ablation_method="zero"
            )
            
            results[neuron_type] = importance_scores
        
        return results
    
    def visualize_neuron_importance(self, results: Dict[str, Any], save_path: str = None):
        """Create comprehensive visualizations of neuron importance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, (neuron_type, importance_data) in enumerate(results.items()):
            row = idx // 2
            col = idx % 2
            
            # Extract importance scores
            neurons = list(importance_data.keys())
            importance_scores = [importance_data[n]['importance_score'] for n in neurons]
            relative_drops = [importance_data[n]['relative_drop'] for n in neurons]
            
            # Plot importance scores
            axes[row, col].bar(neurons, importance_scores, alpha=0.7)
            axes[row, col].set_title(f'{neuron_type.capitalize()} Neuron Importance')
            axes[row, col].set_xlabel('Neuron Index')
            axes[row, col].set_ylabel('Importance Score')
            axes[row, col].grid(True, alpha=0.3)
            
            # Highlight top neurons
            top_neurons = sorted(range(len(importance_scores)), 
                               key=lambda i: importance_scores[i], reverse=True)[:5]
            for top_idx in top_neurons:
                axes[row, col].bar(neurons[top_idx], importance_scores[top_idx], 
                                 color='red', alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("NEURON IMPORTANCE SUMMARY")
        print(f"{'='*60}")
        
        for neuron_type, importance_data in results.items():
            importance_scores = [data['importance_score'] for data in importance_data.values()]
            relative_drops = [data['relative_drop'] for data in importance_data.values()]
            
            print(f"\n{neuron_type.upper()} NEURONS:")
            print(f"  Mean importance: {np.mean(importance_scores):.6f}")
            print(f"  Max importance:  {np.max(importance_scores):.6f}")
            print(f"  Std importance:  {np.std(importance_scores):.6f}")
            print(f"  Mean rel. drop:  {np.mean(relative_drops):.4f}")
            
            # Top 5 most important neurons
            sorted_neurons = sorted(importance_data.items(), 
                                  key=lambda x: x[1]['importance_score'], reverse=True)
            print(f"  Top 5 neurons: {[n for n, _ in sorted_neurons[:5]]}")


def create_synthetic_emotional_audio(emotion="happy", duration=3, sample_rate=16000):
    """Create synthetic emotional audio for testing"""
    t = np.linspace(0, duration, sample_rate * duration)
    
    if emotion == "happy":
        # Happy: higher frequency with vibrato
        vibrato = 1 + 0.1 * np.sin(2 * np.pi * 5 * t)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t * vibrato)
    elif emotion == "sad":
        # Sad: lower frequency, decreasing amplitude
        decay = np.exp(-t * 0.5)
        audio = 0.3 * decay * np.sin(2 * np.pi * 220 * t)
    elif emotion == "angry":
        # Angry: harsh, higher amplitude with noise
        noise = 0.1 * np.random.randn(len(t))
        audio = 0.5 * np.sin(2 * np.pi * 330 * t) + noise
    else:  # neutral
        audio = 0.3 * np.sin(2 * np.pi * 280 * t)
    
    return audio.astype(np.float32)


# Example usage and testing
def test_neuron_ablation():
    """Test the neuron ablation system"""
    print("="*60)
    print("TESTING WAVLM NEURON ABLATION")
    print("="*60)
    
    # Initialize the ablation system
    wavlm_ablator = WavLMNeuronAblation()
    
    # Create test audio
    test_emotions = ["happy", "sad", "angry", "neutral"]
    
    for emotion in test_emotions:
        print(f"\n{'='*40}")
        print(f"TESTING EMOTION: {emotion.upper()}")
        print(f"{'='*40}")
        
        # Create synthetic audio
        audio = create_synthetic_emotional_audio(emotion, duration=2)
        
        try:
            # Analyze neuron importance for this emotion
            results = wavlm_ablator.analyze_emotion_processing_neurons(
                audio_input=audio,
                layer_idx=6,  # Middle layer
                num_neurons=20  # Test 20 neurons for speed
            )
            
            # Visualize results
            wavlm_ablator.visualize_neuron_importance(
                results, 
                save_path=f"neuron_importance_{emotion}.png"
            )
            
            print(f"✓ Successfully analyzed {emotion} emotion")
            
        except Exception as e:
            print(f"✗ Error analyzing {emotion}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_neuron_ablation()