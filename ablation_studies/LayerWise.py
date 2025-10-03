import torch
import torchaudio
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from tqdm import tqdm

class LayerWiseSHAP:
    """
    Layer-by-layer neuron importance analysis using SHAP values for WavLM
    """
    
    def __init__(self, model_name="microsoft/wavlm-base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize WavLM model with SHAP explainer"""
        self.device = device
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Model architecture info
        self.num_layers = len(self.model.encoder.layers)
        self.hidden_size = self.model.config.hidden_size
        
        # Store layer outputs for SHAP analysis
        self.layer_outputs = {}
        self.hooks = []
        
        print(f"WavLM Model Loaded:")
        print(f"  - Layers: {self.num_layers}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Device: {self.device}")
    
    def register_layer_hooks(self):
        """Register hooks to capture intermediate layer outputs"""
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Store layer output [batch, seq_len, hidden_size]
                self.layer_outputs[layer_idx] = hidden_states.detach()
            return hook_fn
        
        # Register hooks for all encoder layers
        for i, layer in enumerate(self.model.encoder.layers):
            hook = layer.register_forward_hook(create_hook(i))
            self.hooks.append(hook)
    
    def remove_layer_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_outputs = {}
    
    def create_model_wrapper(self, target_layer: int, aggregation: str = "mean"):
        """
        Create a wrapper function that extracts features from a specific layer
        
        Args:
            target_layer: Layer index to extract features from
            aggregation: How to aggregate sequence dimension ("mean", "max", "last")
        """
        def model_wrapper(audio_batch):
            """
            Wrapper function for SHAP explainer
            
            Args:
                audio_batch: Batch of audio inputs [batch, seq_len]
            
            Returns:
                Layer features [batch, hidden_size]
            """
            self.layer_outputs = {}  # Clear previous outputs
            
            with torch.no_grad():
                # Convert to tensor if needed
                if isinstance(audio_batch, np.ndarray):
                    audio_batch = torch.tensor(audio_batch, dtype=torch.float32).to(self.device)
                
                # Forward pass to capture layer outputs
                _ = self.model(audio_batch)
                
                if target_layer not in self.layer_outputs:
                    raise ValueError(f"Layer {target_layer} output not captured")
                
                layer_output = self.layer_outputs[target_layer]  # [batch, seq_len, hidden_size]
                
                # Aggregate sequence dimension
                if aggregation == "mean":
                    features = layer_output.mean(dim=1)  # [batch, hidden_size]
                elif aggregation == "max":
                    features = layer_output.max(dim=1)[0]  # [batch, hidden_size]
                elif aggregation == "last":
                    features = layer_output[:, -1, :]  # [batch, hidden_size]
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
                
                return features.cpu().numpy()
        
        return model_wrapper
    
    def create_neuron_wrapper(self, target_layer: int, neuron_idx: int, aggregation: str = "mean"):
        """
        Create wrapper function for single neuron SHAP analysis
        
        Args:
            target_layer: Layer index
            neuron_idx: Neuron index within the layer
            aggregation: How to aggregate sequence dimension
        """
        def neuron_wrapper(audio_batch):
            """Extract single neuron activation"""
            self.layer_outputs = {}
            
            with torch.no_grad():
                # Convert to tensor if needed
                if isinstance(audio_batch, np.ndarray):
                    audio_batch = torch.tensor(audio_batch, dtype=torch.float32).to(self.device)
                
                _ = self.model(audio_batch)
                
                layer_output = self.layer_outputs[target_layer]  # [batch, seq_len, hidden_size]
                neuron_output = layer_output[:, :, neuron_idx]    # [batch, seq_len]
                
                # Aggregate sequence dimension
                if aggregation == "mean":
                    activation = neuron_output.mean(dim=1)  # [batch]
                elif aggregation == "max":
                    activation = neuron_output.max(dim=1)[0]  # [batch]
                elif aggregation == "last":
                    activation = neuron_output[:, -1]  # [batch]
                
                return activation.cpu().numpy().reshape(-1, 1)  # [batch, 1]
        
        return neuron_wrapper
    
    def compute_layer_shap_values(self, 
                                  audio_inputs: torch.Tensor,
                                  target_layer: int,
                                  background_size: int = 10,
                                  test_size: int = 5) -> np.ndarray:
        """
        Compute SHAP values for all neurons in a specific layer
        
        Args:
            audio_inputs: Input audio tensor [batch, seq_len]
            target_layer: Layer to analyze
            background_size: Number of background samples for SHAP
            test_size: Number of test samples
            
        Returns:
            SHAP values [test_samples, hidden_size]
        """
        print(f"\n🔍 Computing SHAP values for Layer {target_layer}...")
        
        # Register hooks to capture layer outputs
        self.register_layer_hooks()
        
        try:
            # Create model wrapper for this layer
            model_wrapper = self.create_model_wrapper(target_layer, aggregation="mean")
            
            # Prepare data
            audio_numpy = audio_inputs.cpu().numpy()
            
            # Select background and test samples
            if len(audio_numpy) < background_size + test_size:
                # If not enough samples, duplicate the input
                audio_numpy = np.tile(audio_numpy, (background_size + test_size, 1))
            
            background_data = audio_numpy[:background_size]
            test_data = audio_numpy[background_size:background_size + test_size]
            
            # Initialize SHAP explainer
            print(f"  Initializing SHAP explainer with {background_size} background samples...")
            explainer = shap.KernelExplainer(model_wrapper, background_data)
            
            # Compute SHAP values
            print(f"  Computing SHAP values for {test_size} test samples...")
            shap_values = explainer.shap_values(test_data, nsamples=100)  # Reduce nsamples for speed
            
            # shap_values shape: [test_samples, hidden_size]
            return shap_values
            
        finally:
            # Always remove hooks
            self.remove_layer_hooks()
    
    def find_important_neurons_shap(self, 
                                   audio_inputs: torch.Tensor,
                                   target_layer: int,
                                   top_k: int = 20) -> Dict[int, Dict[str, float]]:
        """
        Find most important neurons using SHAP values
        
        Args:
            audio_inputs: Input audio tensor
            target_layer: Layer to analyze
            top_k: Number of top neurons to return
            
        Returns:
            Dictionary mapping neuron indices to SHAP-based importance scores
        """
        # Compute SHAP values for the layer
        shap_values = self.compute_layer_shap_values(
            audio_inputs, 
            target_layer,
            background_size=5,  # Reduced for speed
            test_size=3
        )
        
        # Aggregate SHAP values across test samples
        # Use absolute values to capture both positive and negative importance
        abs_shap_values = np.abs(shap_values)
        mean_importance = abs_shap_values.mean(axis=0)  # [hidden_size]
        std_importance = abs_shap_values.std(axis=0)    # [hidden_size]
        
        # Create neuron importance dictionary
        neuron_importance = {}
        
        for neuron_idx in range(len(mean_importance)):
            neuron_importance[neuron_idx] = {
                'shap_importance': float(mean_importance[neuron_idx]),
                'shap_std': float(std_importance[neuron_idx]),
                'shap_raw_values': shap_values[:, neuron_idx].tolist(),
                'layer_idx': target_layer
            }
        
        # Sort by SHAP importance
        sorted_neurons = dict(sorted(neuron_importance.items(), 
                                   key=lambda x: x[1]['shap_importance'], 
                                   reverse=True))
        
        # Return top-k neurons
        top_neurons = dict(list(sorted_neurons.items())[:top_k])
        
        # Print top neurons
        print(f"  Top 5 neurons in layer {target_layer} (SHAP-based):")
        for i, (neuron_idx, info) in enumerate(list(top_neurons.items())[:5]):
            print(f"    #{i+1}: Neuron {neuron_idx} (SHAP importance: {info['shap_importance']:.6f})")
        
        return top_neurons
    
    def progressive_shap_analysis(self,
                                 audio_inputs: torch.Tensor,
                                 layers_to_analyze: List[int] = None,
                                 top_k_per_layer: int = 10) -> Dict[str, Any]:
        """
        Perform progressive SHAP-based analysis across layers
        
        Args:
            audio_inputs: Input audio tensor
            layers_to_analyze: Layers to analyze
            top_k_per_layer: Top neurons per layer
            
        Returns:
            Complete SHAP analysis results
        """
        if layers_to_analyze is None:
            layers_to_analyze = list(range(0, self.num_layers, 2))  # Every 2nd layer
        
        print(f"\n🚀 Starting SHAP-based Layer Analysis")
        print(f"Analyzing layers: {layers_to_analyze}")
        print(f"Top {top_k_per_layer} neurons per layer")
        
        results = {
            'layer_shap_analysis': {},
            'layer_importance_ranking': {},
            'cross_layer_comparison': {}
        }
        
        # Analyze each layer
        for layer_idx in layers_to_analyze:
            print(f"\n{'='*50}")
            print(f"Analyzing Layer {layer_idx}")
            print(f"{'='*50}")
            
            # Get SHAP-based neuron importance
            layer_results = self.find_important_neurons_shap(
                audio_inputs=audio_inputs,
                target_layer=layer_idx,
                top_k=top_k_per_layer
            )
            
            results['layer_shap_analysis'][layer_idx] = layer_results
            
            # Calculate layer-level statistics
            importance_scores = [data['shap_importance'] for data in layer_results.values()]
            layer_stats = {
                'mean_importance': float(np.mean(importance_scores)),
                'max_importance': float(np.max(importance_scores)),
                'std_importance': float(np.std(importance_scores)),
                'top_neuron_idx': max(layer_results.keys(), key=lambda k: layer_results[k]['shap_importance']),
                'layer_idx': layer_idx
            }
            
            results['layer_importance_ranking'][layer_idx] = layer_stats
        
        # Cross-layer comparison
        layer_mean_importance = {
            layer_idx: stats['mean_importance'] 
            for layer_idx, stats in results['layer_importance_ranking'].items()
        }
        
        results['cross_layer_comparison'] = {
            'most_important_layer': max(layer_mean_importance.keys(), 
                                      key=lambda k: layer_mean_importance[k]),
            'layer_importance_ranking': sorted(layer_mean_importance.items(), 
                                             key=lambda x: x[1], reverse=True),
            'importance_distribution': layer_mean_importance
        }
        
        return results
    
    def visualize_shap_analysis(self, results: Dict, save_path: str = None):
        """Create visualizations for SHAP-based analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Layer importance comparison
        layers = sorted(results['layer_importance_ranking'].keys())
        mean_importance = [results['layer_importance_ranking'][l]['mean_importance'] for l in layers]
        max_importance = [results['layer_importance_ranking'][l]['max_importance'] for l in layers]
        
        axes[0, 0].plot(layers, mean_importance, 'b-o', label='Mean SHAP Importance', linewidth=2)
        axes[0, 0].plot(layers, max_importance, 'r-s', label='Max SHAP Importance', linewidth=2)
        axes[0, 0].set_title('SHAP-based Neuron Importance Across Layers')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('SHAP Importance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Top neurons heatmap
        layers_to_show = layers[:min(6, len(layers))]
        shap_matrix = []
        
        for layer_idx in layers_to_show:
            layer_data = results['layer_shap_analysis'][layer_idx]
            importance_scores = [data['shap_importance'] for data in list(layer_data.values())[:10]]
            shap_matrix.append(importance_scores)
        
        if shap_matrix:
            im = axes[0, 1].imshow(shap_matrix, cmap='viridis', aspect='auto')
            axes[0, 1].set_title('Top 10 Neurons SHAP Importance Heatmap')
            axes[0, 1].set_xlabel('Neuron Rank (Top 10)')
            axes[0, 1].set_ylabel('Layer Index')
            axes[0, 1].set_yticks(range(len(layers_to_show)))
            axes[0, 1].set_yticklabels(layers_to_show)
            plt.colorbar(im, ax=axes[0, 1])
        
        # Plot 3: SHAP value distributions
        sample_layer = layers[len(layers)//2]  # Middle layer
        if sample_layer in results['layer_shap_analysis']:
            sample_data = results['layer_shap_analysis'][sample_layer]
            shap_values = []
            for neuron_data in list(sample_data.values())[:20]:  # Top 20 neurons
                shap_values.extend(neuron_data['shap_raw_values'])
            
            axes[1, 0].hist(shap_values, bins=30, alpha=0.7, color='green')
            axes[1, 0].set_title(f'SHAP Values Distribution (Layer {sample_layer})')
            axes[1, 0].set_xlabel('SHAP Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Layer ranking
        layer_ranking = results['cross_layer_comparison']['layer_importance_ranking']
        layer_indices = [item[0] for item in layer_ranking]
        importance_values = [item[1] for item in layer_ranking]
        
        axes[1, 1].bar(range(len(layer_indices)), importance_values, alpha=0.7)
        axes[1, 1].set_title('Layer Importance Ranking (SHAP-based)')
        axes[1, 1].set_xlabel('Layer Rank')
        axes[1, 1].set_ylabel('Mean SHAP Importance')
        axes[1, 1].set_xticks(range(len(layer_indices)))
        axes[1, 1].set_xticklabels([f'L{idx}' for idx in layer_indices], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def load_ravdess_sample(self, ravdess_path: str, emotion: str = "happy") -> Tuple[str, Dict]:
        """
        Load a sample audio file from RAVDESS dataset
        
        Args:
            ravdess_path: Path to RAVDESS dataset
            emotion: Target emotion to analyze
            
        Returns:
            Tuple of (audio_file_path, metadata)
        """
        emotion_map = {
            "neutral": "01", "calm": "02", "happy": "03", "sad": "04",
            "angry": "05", "fearful": "06", "disgust": "07", "surprised": "08"
        }
        
        if emotion not in emotion_map:
            raise ValueError(f"Emotion '{emotion}' not found. Available: {list(emotion_map.keys())}")
        
        emotion_code = emotion_map[emotion]
        
        # Search for files with the emotion code
        ravdess_path = Path(ravdess_path)
        pattern = f"*-*-{emotion_code}-*-*-*-*.wav"
        
        audio_files = list(ravdess_path.rglob(pattern))
        
        if not audio_files:
            raise FileNotFoundError(f"No {emotion} audio files found in {ravdess_path}")
        
        # Select random file
        selected_file = random.choice(audio_files)
        
        # Parse metadata from filename
        parts = selected_file.stem.split('-')
        metadata = {
            'file_path': str(selected_file),
            'emotion': emotion,
            'emotion_code': emotion_code,
            'modality': parts[0],
            'vocal_channel': parts[1],
            'emotion_intensity': parts[3],
            'statement': parts[4],
            'repetition': parts[5],
            'actor': parts[6]
        }
        
        print(f"Selected audio file: {selected_file.name}")
        print(f"Emotion: {emotion} (intensity: {parts[3]})")
        
        return str(selected_file), metadata
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess audio file for WavLM input
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Normalize
        waveform = waveform / waveform.abs().max()
        
        # Feature extraction
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        audio_input = inputs.input_values.to(self.device)
        
        print(f"Audio preprocessed: {audio_input.shape}")
        return audio_input
    
    def save_results(self, results: Dict, metadata: Dict, save_path: str):
        """
        Save analysis results to JSON file
        
        Args:
            results: Analysis results
            metadata: Audio metadata
            save_path: Path to save JSON file
        """
        output_data = {
            'metadata': metadata,
            'analysis_results': results,
            'model_info': {
                'num_layers': self.num_layers,
                'hidden_size': self.hidden_size,
                'device': str(self.device)
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {save_path}")
    
    def progressive_ablation_with_shap_neurons(self,
                                             audio_inputs: torch.Tensor,
                                             shap_results: Dict,
                                             task_metric_fn: callable) -> Dict[str, Any]:
        """
        Perform progressive ablation using SHAP-identified important neurons
        
        Args:
            audio_inputs: Input audio tensor
            shap_results: Results from SHAP analysis
            task_metric_fn: Function to evaluate model performance
            
        Returns:
            Progressive ablation results
        """
        print(f"\n🎯 Progressive Ablation with SHAP-identified Neurons")
        
        # Get baseline performance
        baseline_score = task_metric_fn(self.model, audio_inputs)
        print(f"Baseline performance: {baseline_score:.4f}")
        
        ablation_results = {
            'baseline_score': baseline_score,
            'progressive_scores': [],
            'cumulative_neurons_ablated': []
        }
        
        # Collect all important neurons from SHAP analysis
        all_important_neurons = []
        for layer_idx, layer_data in shap_results['layer_shap_analysis'].items():
            for neuron_idx, neuron_info in layer_data.items():
                all_important_neurons.append({
                    'layer_idx': layer_idx,
                    'neuron_idx': neuron_idx,
                    'shap_importance': neuron_info['shap_importance']
                })
        
        # Sort by SHAP importance globally
        all_important_neurons.sort(key=lambda x: x['shap_importance'], reverse=True)
        
        # Progressive ablation
        neurons_to_ablate = []
        
        for i, neuron_info in enumerate(all_important_neurons[:20]):  # Top 20 globally
            neurons_to_ablate.append(neuron_info)
            
            # Perform ablation
            score = self.ablate_neurons_and_evaluate(
                audio_inputs, 
                neurons_to_ablate, 
                task_metric_fn
            )
            
            ablation_results['progressive_scores'].append(score)
            ablation_results['cumulative_neurons_ablated'].append(len(neurons_to_ablate))
            
            print(f"  Step {i+1}: Ablated {len(neurons_to_ablate)} neurons, Score: {score:.4f}")
        
        return ablation_results
    
    def ablate_neurons_and_evaluate(self, 
                                   audio_inputs: torch.Tensor,
                                   neurons_to_ablate: List[Dict],
                                   task_metric_fn: callable) -> float:
        """
        Ablate specific neurons and evaluate performance
        
        Args:
            audio_inputs: Input audio tensor
            neurons_to_ablate: List of neurons to ablate
            task_metric_fn: Evaluation function
            
        Returns:
            Performance score after ablation
        """
        # Group neurons by layer for efficient ablation
        layer_neurons = {}
        for neuron_info in neurons_to_ablate:
            layer_idx = neuron_info['layer_idx']
            neuron_idx = neuron_info['neuron_idx']
            
            if layer_idx not in layer_neurons:
                layer_neurons[layer_idx] = []
            layer_neurons[layer_idx].append(neuron_idx)
        
        # Create ablation hooks
        hooks = []
        
        def create_ablation_hook(layer_idx, neurons_to_zero):
            def ablation_hook(module, input_tensor, output_tensor):
                if isinstance(output_tensor, tuple):
                    hidden_states = output_tensor[0].clone()
                else:
                    hidden_states = output_tensor.clone()
                
                # Ablate specified neurons
                for neuron_idx in neurons_to_zero:
                    hidden_states[:, :, neuron_idx] = 0.0
                
                if isinstance(output_tensor, tuple):
                    return (hidden_states,) + output_tensor[1:]
                else:
                    return hidden_states
            
            return ablation_hook
        
        # Register ablation hooks
        for layer_idx, neuron_indices in layer_neurons.items():
            hook = self.model.encoder.layers[layer_idx].register_forward_hook(
                create_ablation_hook(layer_idx, neuron_indices)
            )
            hooks.append(hook)
        
        try:
            # Evaluate with ablated neurons
            score = task_metric_fn(self.model, audio_inputs)
            return score
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()


def run_shap_based_analysis(ravdess_path: str, 
                           emotion: str = "happy",
                           save_dir: str = "shap_results"):
    """
    Run SHAP-based neuron importance analysis
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("🎵 WAVLM SHAP-BASED NEURON ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = LayerWiseSHAP()
    
    # Load and preprocess audio
    audio_path, metadata = analyzer.load_ravdess_sample(ravdess_path, emotion)
    audio_input = analyzer.preprocess_audio(audio_path)
    
    # Run SHAP analysis
    key_layers = [0, 2, 4, 6, 8, 10, 11]
    results = analyzer.progressive_shap_analysis(
        audio_inputs=audio_input,
        layers_to_analyze=key_layers,
        top_k_per_layer=15
    )
    
    # Visualize and save
    viz_path = os.path.join(save_dir, f"shap_analysis_{emotion}.png")
    analyzer.visualize_shap_analysis(results, save_path=viz_path)
    
    results_path = os.path.join(save_dir, f"shap_analysis_{emotion}.json")
    analyzer.save_results(results, metadata, results_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SHAP ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    most_important_layer = results['cross_layer_comparison']['most_important_layer']
    print(f"Most important layer: {most_important_layer}")
    
    print("\nLayer importance ranking:")
    for i, (layer_idx, importance) in enumerate(results['cross_layer_comparison']['layer_importance_ranking'][:5]):
        print(f"  #{i+1}: Layer {layer_idx} (SHAP importance: {importance:.6f})")
    
    return results, metadata


# Example task metric function for emotion classification
def emotion_task_metric(model, audio_input):
    """
    Example task metric function - replace with your specific task
    
    Args:
        model: WavLM model
        audio_input: Input audio tensor
        
    Returns:
        Performance score (higher is better)
    """
    with torch.no_grad():
        outputs = model(audio_input)
        # Example: use mean of last hidden state as emotion representation
        hidden_states = outputs.last_hidden_state.mean(dim=1)
        # Return a dummy score - replace with actual emotion classification metric
        return float(hidden_states.norm().cpu())


# Example usage
if __name__ == "__main__":
    # Example usage
    ravdess_path = "path/to/ravdess/dataset"
    
    try:
        results, metadata = run_shap_based_analysis(
            ravdess_path=ravdess_path,
            emotion="happy",
            save_dir="shap_results"
        )
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure RAVDESS dataset is available and path is correct")