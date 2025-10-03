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
import warnings

# Add missing imports for enhanced SHAP functionality
try:
    import shap
    from scipy.stats import spearmanr
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP or scipy not available. Enhanced analysis features will be disabled.")

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
        """
        # RAVDESS emotion mapping
        emotion_map = {
            1: "neutral", 2: "calm", 3: "happy", 4: "sad",
            5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
        }
        
        audio_files = []
        ravdess_path = Path(ravdess_path)
        
        for actor_folder in ravdess_path.glob("Actor_*"):
            if not actor_folder.is_dir():
                continue
                
            actor_num = int(actor_folder.name.split("_")[1])
            
            if actor_id is not None and actor_num != actor_id:
                continue
            
            for audio_file in actor_folder.glob("*.wav"):
                parts = audio_file.stem.split('-')
                
                if len(parts) >= 7:
                    emotion_id = int(parts[2])
                    emotion_name = emotion_map.get(emotion_id, "unknown")
                    
                    if emotion_label is not None and emotion_name != emotion_label:
                        continue
                    
                    metadata = {
                        'file_path': str(audio_file),
                        'emotion_id': emotion_id,
                        'emotion_name': emotion_name,
                        'intensity': int(parts[3]),
                        'statement': int(parts[4]),
                        'repetition': int(parts[5]),
                        'actor_id': int(parts[6]),
                        'gender': 'female' if int(parts[6]) % 2 == 0 else 'male'
                    }
                    
                    audio_files.append((str(audio_file), metadata))
        
        if not audio_files:
            raise ValueError(f"No RAVDESS files found for emotion='{emotion_label}', actor='{actor_id}'")
        
        audio_path, metadata = random.choice(audio_files)
        print(f"Selected audio: {metadata['emotion_name']} emotion, Actor {metadata['actor_id']} ({metadata['gender']})")
        
        return audio_path, metadata
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """Load and preprocess audio file for WavLM"""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=target_sr, 
            return_tensors="pt"
        )
        
        return inputs["input_values"].to(self.device)
    
    def create_task_metric(self, task_type: str = "emotion_representation"):
        """Create task-specific metric function with multiple options including SHAP XAI"""
        
        if task_type == "emotion_representation":
            def emotion_metric(model, audio_input):
                """Measure quality of emotion representation"""
                with torch.no_grad():
                    outputs = model(audio_input)
                    emotion_repr = outputs.last_hidden_state.mean(dim=1)
                    return emotion_repr.norm(dim=1).mean().item()
            return emotion_metric
            
        elif task_type == "hidden_state_variance":
            def variance_metric(model, audio_input):
                """Measure variance in hidden states (diversity metric)"""
                with torch.no_grad():
                    outputs = model(audio_input)
                    hidden_states = outputs.last_hidden_state
                    variance = hidden_states.var(dim=1).mean().item()
                    return variance
            return variance_metric
        
        elif task_type == "cosine_similarity":
            def cosine_metric(model, audio_input):
                with torch.no_grad():
                    outputs = model(audio_input)
                    emotion_repr = outputs.last_hidden_state.mean(dim=1)
                    if hasattr(self, 'emotion_prototypes'):
                        prototype = self.emotion_prototypes.get("happy", torch.zeros_like(emotion_repr))
                        similarity = torch.cosine_similarity(emotion_repr, prototype, dim=1)
                        return similarity.mean().item()
                    else:
                        return emotion_repr.norm(dim=1).mean().item()
            return cosine_metric
        
        elif task_type == "representation_entropy":
            def entropy_metric(model, audio_input):
                with torch.no_grad():
                    outputs = model(audio_input)
                    emotion_repr = outputs.last_hidden_state.mean(dim=1)
                    probs = torch.softmax(emotion_repr, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    return entropy.mean().item()
            return entropy_metric
        
        elif task_type == "attention_concentration":
            def attention_metric(model, audio_input):
                with torch.no_grad():
                    outputs = model(audio_input, output_attentions=True)
                    if hasattr(outputs, 'attentions') and outputs.attentions:
                        attention_weights = outputs.attentions[-1]
                        entropy = -torch.sum(
                            attention_weights * torch.log(attention_weights + 1e-8), 
                            dim=-1
                        ).mean().item()
                        return -entropy
                    else:
                        emotion_repr = outputs.last_hidden_state.mean(dim=1)
                        return emotion_repr.norm(dim=1).mean().item()
            return attention_metric
        
        elif task_type == "shap_neuron_importance":
            def shap_metric(model, audio_input):
                """Use SHAP to measure overall model importance"""
                if not SHAP_AVAILABLE:
                    print("SHAP not available, falling back to L2 norm")
                    with torch.no_grad():
                        outputs = model(audio_input)
                        emotion_repr = outputs.last_hidden_state.mean(dim=1)
                        return emotion_repr.norm(dim=1).mean().item()
                
                try:
                    # Use GradientExplainer instead of DeepExplainer for better compatibility
                    def model_forward(x):
                        outputs = model(x)
                        emotion_repr = outputs.last_hidden_state.mean(dim=1)
                        return emotion_repr.norm(dim=1).mean()
                    
                    # Create a simpler explainer
                    explainer = shap.GradientExplainer(model, audio_input)
                    shap_values = explainer.shap_values(audio_input)
                    return torch.tensor(shap_values).abs().mean().item()
                    
                except Exception as e:
                    print(f"SHAP GradientExplainer failed: {e}, trying alternative approach")
                    try:
                        # Alternative: Use model directly with SHAP
                        background = torch.zeros_like(audio_input)
                        explainer = shap.DeepExplainer(model, background)
                        shap_values = explainer.shap_values(audio_input)
                        return torch.tensor(shap_values).abs().mean().item()
                    except Exception as e2:
                        print(f"All SHAP methods failed: {e2}, falling back to L2 norm")
                        with torch.no_grad():
                            outputs = model(audio_input)
                            emotion_repr = outputs.last_hidden_state.mean(dim=1)
                            return emotion_repr.norm(dim=1).mean().item()
            return shap_metric
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def find_important_neurons_in_layer(self, 
                                      audio_input: torch.Tensor,
                                      layer_idx: int,
                                      task_metric_fn,
                                      num_neurons_to_test: int = 50,
                                      ablation_method: str = "zero") -> Dict[int, Dict[str, float]]:
        """Find most important neurons in a specific layer using ablation"""
        print(f"\n🔍 Analyzing Layer {layer_idx} ({num_neurons_to_test} neurons)...")
        
        # Add memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        baseline_score = task_metric_fn(self.model, audio_input)
        print(f"  Baseline score: {baseline_score:.6f}")
        
        neuron_importance = {}
        layer = self.model.encoder.layers[layer_idx]
        
        for neuron_idx in tqdm(range(min(num_neurons_to_test, self.hidden_size)), 
                              desc=f"Layer {layer_idx} neurons"):
            
            # Clear cache periodically
            if neuron_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            def create_ablation_hook(neuron_to_ablate):
                def ablation_hook(module, input_tensor, output_tensor):
                    if isinstance(output_tensor, tuple):
                        hidden_states = output_tensor[0].clone()
                        other_outputs = output_tensor[1:]
                    else:
                        hidden_states = output_tensor.clone()
                        other_outputs = ()
                    
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
                    
                    if other_outputs:
                        return (hidden_states,) + other_outputs
                    return hidden_states
                
                return ablation_hook
            
            hook = layer.register_forward_hook(create_ablation_hook(neuron_idx))
            
            try:
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
                hook.remove()
        
        sorted_neurons = dict(sorted(neuron_importance.items(), 
                                   key=lambda x: x[1]['importance_score'], 
                                   reverse=True))
        
        print(f"  Top 5 neurons in layer {layer_idx}:")
        for i, (neuron_idx, info) in enumerate(list(sorted_neurons.items())[:5]):
            print(f"    #{i+1}: Neuron {neuron_idx} (importance: {info['importance_score']:.6f})")
        
        return sorted_neurons

    def get_shap_top_neurons_per_layer(self, audio_input: torch.Tensor, 
                                     layer_idx: int, 
                                     top_k: int = 10) -> List[Tuple[int, float]]:
        """Use SHAP to identify top neurons in a specific layer - FIXED VERSION"""
        if not SHAP_AVAILABLE:
            print("SHAP not available, falling back to random neuron selection")
            neurons_to_test = min(top_k, self.hidden_size)
            random_neurons = random.sample(range(neurons_to_test), min(top_k, neurons_to_test))
            return [(n, random.random()) for n in random_neurons]
        
        # Create a custom PyTorch module for SHAP compatibility
        class NeuronExtractor(torch.nn.Module):
            def __init__(self, base_model, layer_idx, neuron_idx):
                super().__init__()
                self.base_model = base_model
                self.layer_idx = layer_idx
                self.neuron_idx = neuron_idx
                
            def forward(self, x):
                with torch.no_grad():
                    outputs = self.base_model(x, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[self.layer_idx]
                    neuron_activation = hidden_states[:, :, self.neuron_idx].mean(dim=1)
                    return neuron_activation
        
        neuron_importance = {}
        neurons_to_test = min(50, self.hidden_size)
        
        print(f"\n🔍 SHAP Analysis for Layer {layer_idx} (testing {neurons_to_test} neurons)...")
        
        for neuron_idx in tqdm(range(neurons_to_test), desc=f"SHAP Layer {layer_idx}"):
            # Clear cache periodically
            if neuron_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            try:
                # Create neuron extractor module
                neuron_extractor = NeuronExtractor(self.model, layer_idx, neuron_idx)
                
                # Use GradientExplainer for better compatibility
                background = torch.zeros_like(audio_input)
                explainer = shap.GradientExplainer(neuron_extractor, background)
                
                # Get SHAP values
                shap_values = explainer.shap_values(audio_input)
                importance = torch.tensor(shap_values).abs().mean().item()
                neuron_importance[neuron_idx] = importance
                
            except Exception as e:
                print(f"    SHAP failed for neuron {neuron_idx}: {e}")
                # Fallback to gradient-based importance
                try:
                    audio_input.requires_grad_(True)
                    outputs = self.model(audio_input, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[layer_idx]
                    neuron_activation = hidden_states[:, :, neuron_idx].mean()
                    
                    # Compute gradient
                    neuron_activation.backward(retain_graph=True)
                    gradient_importance = audio_input.grad.abs().mean().item()
                    neuron_importance[neuron_idx] = gradient_importance
                    
                    # Reset gradients
                    audio_input.grad = None
                    audio_input.requires_grad_(False)
                    
                except Exception as e2:
                    print(f"    Gradient fallback also failed for neuron {neuron_idx}: {e2}")
                    neuron_importance[neuron_idx] = 0.0
        
        sorted_neurons = sorted(neuron_importance.items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:top_k]
        
        print(f"  Top 5 SHAP neurons in layer {layer_idx}:")
        for i, (neuron_idx, importance) in enumerate(sorted_neurons[:5]):
            print(f"    #{i+1}: Neuron {neuron_idx} (SHAP importance: {importance:.6f})")
        
        return sorted_neurons

    def compare_ablation_vs_shap_neurons(self, 
                                       audio_input: torch.Tensor, 
                                       layer_idx: int,
                                       task_metric_fn,
                                       top_k: int = 10) -> Dict[str, Any]:
        """Compare neuron importance rankings from ablation study vs SHAP"""
        if not SHAP_AVAILABLE:
            print(f"SHAP not available, skipping comparison for Layer {layer_idx}")
            return {'error': 'SHAP not available'}
        
        print(f"\n🔄 Comparing Ablation vs SHAP for Layer {layer_idx}")
        
        # Get ablation-based top neurons
        print("Running ablation analysis...")
        ablation_results = self.find_important_neurons_in_layer(
            audio_input=audio_input,
            layer_idx=layer_idx,
            task_metric_fn=task_metric_fn,
            num_neurons_to_test=50
        )
        
        ablation_top = list(ablation_results.keys())[:top_k]
        
        # Get SHAP-based top neurons
        print("Running SHAP analysis...")
        shap_top = self.get_shap_top_neurons_per_layer(
            audio_input=audio_input,
            layer_idx=layer_idx,
            top_k=top_k
        )
        
        shap_neuron_indices = [neuron_idx for neuron_idx, _ in shap_top]
        
        # Calculate overlap
        overlap = set(ablation_top) & set(shap_neuron_indices)
        overlap_percentage = len(overlap) / top_k * 100
        
        # Rank correlation
        common_neurons = list(overlap)
        if len(common_neurons) > 2 and SHAP_AVAILABLE:
            ablation_ranks = [ablation_top.index(n) for n in common_neurons]
            shap_ranks = [shap_neuron_indices.index(n) for n in common_neurons]
            correlation, p_value = spearmanr(ablation_ranks, shap_ranks)
        else:
            correlation, p_value = 0.0, 1.0
        
        results = {
            'layer_idx': layer_idx,
            'ablation_top_neurons': ablation_top,
            'shap_top_neurons': shap_neuron_indices,
            'overlap_neurons': list(overlap),
            'overlap_percentage': overlap_percentage,
            'rank_correlation': correlation,
            'correlation_p_value': p_value,
            'agreement_level': 'High' if overlap_percentage > 70 else 'Medium' if overlap_percentage > 40 else 'Low'
        }
        
        print(f"\n📊 Comparison Results for Layer {layer_idx}:")
        print(f"  Overlap: {len(overlap)}/{top_k} neurons ({overlap_percentage:.1f}%)")
        print(f"  Rank correlation: {correlation:.3f} (p={p_value:.3f})")
        print(f"  Agreement level: {results['agreement_level']}")
        print(f"  Common important neurons: {list(overlap)}")
        
        return results

    def enhanced_progressive_layer_ablation(self,
                                          audio_input: torch.Tensor,
                                          task_metric_fn,
                                          layers_to_analyze: List[int] = None,
                                          top_k_per_layer: int = 5,
                                          use_shap: bool = False,
                                          comparison_mode: bool = False) -> Dict[int, Any]:
        """
        Enhanced progressive ablation with SHAP integration
        """
        if layers_to_analyze is None:
            layers_to_analyze = list(range(self.num_layers))
        
        print(f"\n🚀 Enhanced Progressive Layer Analysis")
        print(f"Method: {'SHAP + Ablation' if use_shap else 'Ablation Only'}")
        print(f"Comparison mode: {comparison_mode}")
        
        results = {
            'baseline_score': task_metric_fn(self.model, audio_input),
            'method': 'shap+ablation' if use_shap else 'ablation',
            'layer_analysis': {},
            'shap_comparisons': {} if comparison_mode else None,
            'progressive_ablation': {},
            'cumulative_effects': []
        }
        
        # Individual layer analysis
        for layer_idx in layers_to_analyze:
            
            if comparison_mode:
                comparison_result = self.compare_ablation_vs_shap_neurons(
                    audio_input=audio_input,
                    layer_idx=layer_idx,
                    task_metric_fn=task_metric_fn,
                    top_k=top_k_per_layer * 2
                )
                results['shap_comparisons'][layer_idx] = comparison_result
                
                # Use ablation results for progressive analysis
                layer_results = self.find_important_neurons_in_layer(
                    audio_input=audio_input,
                    layer_idx=layer_idx,
                    task_metric_fn=task_metric_fn,
                    num_neurons_to_test=50
                )
                
            elif use_shap and SHAP_AVAILABLE:
                shap_neurons = self.get_shap_top_neurons_per_layer(
                    audio_input=audio_input,
                    layer_idx=layer_idx,
                    top_k=top_k_per_layer * 2
                )
                
                # Convert to ablation format for consistency
                layer_results = {}
                for i, (neuron_idx, shap_importance) in enumerate(shap_neurons):
                    layer_results[neuron_idx] = {
                        'importance_score': shap_importance,
                        'method': 'shap',
                        'layer_idx': layer_idx,
                        'rank': i + 1
                    }
            else:
                # Standard ablation analysis (fallback for all cases)
                layer_results = self.find_important_neurons_in_layer(
                    audio_input=audio_input,
                    layer_idx=layer_idx,
                    task_metric_fn=task_metric_fn,
                    num_neurons_to_test=50
                )
            
            results['layer_analysis'][layer_idx] = layer_results
    
        # Progressive ablation (same as original)
        active_hooks = []
        cumulative_ablated_layers = []
        
        for layer_idx in sorted(layers_to_analyze):
            print(f"\n📍 Adding Layer {layer_idx} to progressive ablation...")
            
            layer_neurons = results['layer_analysis'][layer_idx]
            top_neurons = list(layer_neurons.keys())[:top_k_per_layer]
            
            layer = self.model.encoder.layers[layer_idx]
            
            def create_progressive_hook(neurons_to_ablate):
                def progressive_hook(module, input_tensor, output_tensor):
                    if isinstance(output_tensor, tuple):
                        hidden_states = output_tensor[0].clone()
                        other_outputs = output_tensor[1:]
                    else:
                        hidden_states = output_tensor.clone()
                        other_outputs = ()
                    
                    for neuron_idx in neurons_to_ablate:
                        hidden_states[:, :, neuron_idx] = 0.0
                    
                    if other_outputs:
                        return (hidden_states,) + other_outputs
                    return hidden_states
                
                return progressive_hook
            
            hook = layer.register_forward_hook(create_progressive_hook(top_neurons))
            active_hooks.append((layer_idx, hook))
            cumulative_ablated_layers.append(layer_idx)
            
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
        
        # Cleanup
        for layer_idx, hook in active_hooks:
            hook.remove()
        
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
        
    
    def visualize_layer_analysis(self, results: Dict, save_path: str = None):
        """Create comprehensive visualizations of layer-wise analysis"""
        
        # Add validation
        if not results or 'layer_analysis' not in results:
            print("Warning: No layer analysis results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data for plotting
        layers = sorted(results['layer_analysis'].keys())
        
        if not layers:
            print("Warning: No layers found in results")
            plt.close(fig)
            return
        
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


def run_complete_layer_wise_analysis(ravdess_path: str, 
                                   emotion: str = "happy",
                                   actor_id: int = None,
                                   save_dir: str = "results",
                                   use_enhanced_analysis: bool = False,
                                   task_metric_type: str = "emotion_representation"):
    """
    Run complete layer-wise neuron ablation analysis with optional SHAP integration
    """
    # Add error handling for path validation
    if not os.path.exists(ravdess_path):
        raise FileNotFoundError(f"RAVDESS path not found: {ravdess_path}")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n🎭 Starting Layer-wise Neuron Ablation Analysis")
    print(f"Emotion: {emotion}")
    print(f"Enhanced analysis: {use_enhanced_analysis}")
    print(f"Task metric: {task_metric_type}")
    
    # Step 1: Initialize analyzer
    print("Step 1: Initializing WavLM analyzer...")
    analyzer = LayerWiseNeuronAblation()
    
    # Step 2: Load RAVDESS sample
    print("Step 2: Loading RAVDESS sample...")
    audio_path, metadata = analyzer.load_ravdess_sample(
        ravdess_path=ravdess_path,
        emotion_label=emotion,
        actor_id=actor_id
    )
    
    # Step 3: Preprocess audio
    print("Step 3: Preprocessing audio...")
    audio_input = analyzer.preprocess_audio(audio_path)
    
    # Step 4: Create task metric (now with more options)
    print(f"Step 4: Creating task metric ({task_metric_type})...")
    task_metric = analyzer.create_task_metric(task_metric_type)
    
    # Step 5: Run analysis (choose enhanced or standard)
    print("Step 5: Running layer-wise neuron ablation...")
    key_layers = [0, 2, 4, 6, 8, 10, 11]
    
    if use_enhanced_analysis:
        # Use enhanced analysis with SHAP
        results = analyzer.enhanced_progressive_layer_ablation(
            audio_input=audio_input,
            task_metric_fn=task_metric,
            layers_to_analyze=key_layers,
            top_k_per_layer=5,
            use_shap=True,
            comparison_mode=True  # Compare both methods
        )
    else:
        # Use standard analysis (enhanced method without SHAP)
        results = analyzer.enhanced_progressive_layer_ablation(
            audio_input=audio_input,
            task_metric_fn=task_metric,
            layers_to_analyze=key_layers,
            top_k_per_layer=5,
            use_shap=False,
            comparison_mode=False
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


if __name__ == "__main__":
    RAVDESS_PATH = r"/kaggle/input/ravdess-emotional-speech-audio"  # ⚠️ UPDATE THIS PATH
    
    # Check if RAVDESS path exists
    if not os.path.exists(RAVDESS_PATH):
        print(f"❌ Error: RAVDESS path not found: {RAVDESS_PATH}")
        print("Please update the RAVDESS_PATH variable with the correct path.")
        exit(1)
    
    # Test different analysis methods
    analysis_configs = [
        {"method": "standard", "task_metric": "emotion_representation", "use_enhanced": False},
        {"method": "enhanced_shap", "task_metric": "shap_neuron_importance", "use_enhanced": True},
        {"method": "entropy_based", "task_metric": "representation_entropy", "use_enhanced": False},
        {"method": "attention_based", "task_metric": "attention_concentration", "use_enhanced": False}
    ]
    
    emotions_to_test = ["happy"]
    
    for emotion in emotions_to_test:
        for config in analysis_configs:
            print(f"\n{'🎭'*20}")
            print(f"ANALYZING EMOTION: {emotion.upper()} - METHOD: {config['method'].upper()}")
            print(f"{'🎭'*20}")
            
            try:
                results, metadata = run_complete_layer_wise_analysis(
                    ravdess_path=RAVDESS_PATH,
                    emotion=emotion,
                    save_dir=f"results_{emotion}_{config['method']}",
                    use_enhanced_analysis=config['use_enhanced'],
                    task_metric_type=config['task_metric']
                )
                print(f"✅ Successfully analyzed {emotion} with {config['method']}")
                
                # Memory cleanup after each analysis
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error analyzing {emotion} with {config['method']}: {e}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                continue
    
    print(f"\n🎉 Multi-method analysis complete!")