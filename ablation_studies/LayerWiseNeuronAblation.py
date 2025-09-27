#!/usr/bin/env python3

import torch
import torchaudio
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor  # ✅ Fixed import
from captum.attr._core.feature_ablation import FeatureAblation
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from tqdm import tqdm

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
    
    # Step 5: Run layer-wise analysis
    print("Step 5: Running layer-wise neuron ablation...")
    
    # Analyze key layers (not all to save time)
    key_layers = [0, 2, 4, 6, 8, 10, 11]  # Early, middle, and late layers
    
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
    RAVDESS_PATH = r"C:\path\to\your\ravdess\dataset"  # ⚠️ UPDATE THIS PATH
    
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