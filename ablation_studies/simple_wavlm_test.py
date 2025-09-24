#!/usr/bin/env python3

"""
Simplified WavLM Neuron Analysis Test (without Captum dependency)
This demonstrates the core WavLM analysis without the complex ablation parts
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import WavLMModel, Wav2Vec2Processor
import warnings
warnings.filterwarnings('ignore')

class SimpleWavLMAnalyzer:
    """
    Simplified WavLM analyzer that focuses on neuron activations without ablation
    """
    
    def __init__(self, model_name="microsoft/wavlm-base-plus"):
        """Initialize WavLM model"""
        print(f"Loading WavLM model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model directly without processor to avoid tokenizer issues
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.num_layers = len(self.model.encoder.layers)
        print(f"Model loaded with {self.num_layers} transformer layers")
    
    def create_synthetic_audio(self, duration=3.0, sample_rate=16000, emotion_type="happy"):
        """Create synthetic emotion-like audio"""
        print(f"Creating synthetic {emotion_type} audio ({duration}s)")
        
        t = torch.linspace(0, duration, int(duration * sample_rate))
        
        if emotion_type == "happy":
            # Higher frequency, more variation
            signal = (torch.sin(2 * np.pi * 150 * t) * torch.exp(-t * 0.5) + 
                     0.3 * torch.sin(2 * np.pi * 300 * t) +
                     0.1 * torch.randn(len(t)))
        elif emotion_type == "sad":
            # Lower frequencies, less variation
            signal = (torch.sin(2 * np.pi * 100 * t) * torch.exp(-t * 0.3) + 
                     0.2 * torch.sin(2 * np.pi * 200 * t) +
                     0.05 * torch.randn(len(t)))
        elif emotion_type == "angry":
            # Sharp, harsh components
            signal = (torch.sin(2 * np.pi * 200 * t) * (1 + 0.5 * torch.sin(2 * np.pi * 10 * t)) +
                     0.4 * torch.sign(torch.sin(2 * np.pi * 250 * t)) +
                     0.2 * torch.randn(len(t)))
        else:
            # Neutral
            signal = torch.sin(2 * np.pi * 120 * t) + 0.1 * torch.randn(len(t))
        
        # Normalize
        signal = signal / torch.max(torch.abs(signal)) * 0.8
        return signal.unsqueeze(0)  # Add batch dimension
    
    def preprocess_audio(self, audio_tensor, sample_rate=16000):
        """Preprocess audio for WavLM"""
        # Simple preprocessing - normalize and convert to proper format
        audio_np = audio_tensor.squeeze().numpy()
        
        # Normalize audio to [-1, 1] range
        if audio_np.max() != 0:
            audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))
        
        # Convert back to tensor and add batch dimension
        audio_tensor = torch.FloatTensor(audio_np).unsqueeze(0).to(self.device)
        
        return audio_tensor
    
    def analyze_neuron_activations(self, audio_input, target_layers=None):
        """
        Analyze neuron activations across layers
        
        Args:
            audio_input: Preprocessed audio tensor
            target_layers: List of layer indices to analyze (default: [2, 6, 10])
        
        Returns:
            Dictionary with layer activations
        """
        if target_layers is None:
            target_layers = [2, 6, 10]
        
        print(f"Analyzing activations in layers: {target_layers}")
        
        # Hook to capture layer outputs
        activations = {}
        
        def create_hook(layer_name):
            def hook(module, input, output):
                # output is a tuple, get the hidden states
                if isinstance(output, tuple):
                    activations[layer_name] = output[0].detach()
                else:
                    activations[layer_name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        for layer_idx in target_layers:
            layer = self.model.encoder.layers[layer_idx]
            hook = layer.register_forward_hook(create_hook(f"layer_{layer_idx}"))
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(audio_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def analyze_specific_neurons(self, activations, layer_name, neuron_indices=None):
        """
        Analyze specific neurons in a layer
        
        Args:
            activations: Dictionary of layer activations
            layer_name: Name of the layer to analyze
            neuron_indices: List of neuron indices to focus on
        
        Returns:
            Statistics about neuron activations
        """
        if layer_name not in activations:
            print(f"Layer {layer_name} not found in activations")
            return None
        
        layer_output = activations[layer_name]  # Shape: [batch, seq_len, hidden_dim]
        
        if neuron_indices is None:
            neuron_indices = [64, 128, 256, 384, 512]  # Default neurons to analyze
        
        # Filter valid indices
        max_neurons = layer_output.shape[-1]
        valid_indices = [idx for idx in neuron_indices if idx < max_neurons]
        
        print(f"Analyzing neurons {valid_indices} in {layer_name}")
        print(f"Layer output shape: {layer_output.shape}")
        
        neuron_stats = {}
        
        for neuron_idx in valid_indices:
            neuron_activations = layer_output[0, :, neuron_idx]  # [seq_len]
            
            stats = {
                'mean_activation': neuron_activations.mean().item(),
                'max_activation': neuron_activations.max().item(),
                'min_activation': neuron_activations.min().item(),
                'std_activation': neuron_activations.std().item(),
                'peak_time': neuron_activations.argmax().item(),
                'activations': neuron_activations.cpu().numpy()
            }
            
            neuron_stats[f'neuron_{neuron_idx}'] = stats
        
        return neuron_stats
    
    def visualize_neuron_analysis(self, audio_tensor, neuron_stats_dict, save_path=None):
        """
        Visualize neuron analysis results
        
        Args:
            audio_tensor: Original audio tensor
            neuron_stats_dict: Dictionary of neuron statistics by layer
            save_path: Path to save the plot
        """
        num_layers = len(neuron_stats_dict)
        fig, axes = plt.subplots(num_layers + 1, 1, figsize=(15, 4 * (num_layers + 1)))
        
        if num_layers == 0:
            axes = [axes]
        
        # Plot original audio
        audio_np = audio_tensor.squeeze().cpu().numpy()
        time_audio = np.linspace(0, len(audio_np) / 16000, len(audio_np))
        
        axes[0].plot(time_audio, audio_np)
        axes[0].set_title("Original Audio Waveform")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        
        # Plot neuron activations for each layer
        layer_idx = 1
        for layer_name, neuron_stats in neuron_stats_dict.items():
            ax = axes[layer_idx]
            
            # Create time axis for activations (assuming they match audio length proportionally)
            for neuron_name, stats in neuron_stats.items():
                activations = stats['activations']
                time_act = np.linspace(0, len(audio_np) / 16000, len(activations))
                
                ax.plot(time_act, activations, label=f"{neuron_name} (mean: {stats['mean_activation']:.3f})")
            
            ax.set_title(f"{layer_name} - Neuron Activations")
            ax.set_ylabel("Activation")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if layer_idx == num_layers:  # Last subplot
                ax.set_xlabel("Time (seconds)")
            
            layer_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()

def test_basic_wavlm_analysis():
    """Test basic WavLM analysis without ablation"""
    print("🚀 Testing Basic WavLM Neuron Analysis")
    print("="*50)
    
    # Create analyzer
    analyzer = SimpleWavLMAnalyzer()
    
    # Create test results directory
    results_dir = Path("simple_test_results")
    results_dir.mkdir(exist_ok=True)
    
    emotions = ["happy", "sad", "angry", "neutral"]
    all_results = {}
    
    for emotion in emotions:
        print(f"\n📊 Analyzing {emotion} emotion...")
        
        # Create synthetic audio
        audio_tensor = analyzer.create_synthetic_audio(duration=2.0, emotion_type=emotion)
        
        # Preprocess for WavLM
        audio_input = analyzer.preprocess_audio(audio_tensor)
        
        # Analyze neuron activations
        activations = analyzer.analyze_neuron_activations(audio_input, target_layers=[2, 6, 10])
        
        # Analyze specific neurons in each layer
        emotion_results = {}
        for layer_name in activations.keys():
            neuron_stats = analyzer.analyze_specific_neurons(
                activations, layer_name, neuron_indices=[64, 128, 256, 384]
            )
            emotion_results[layer_name] = neuron_stats
        
        all_results[emotion] = emotion_results
        
        # Visualize results for this emotion
        save_path = results_dir / f"wavlm_analysis_{emotion}.png"
        analyzer.visualize_neuron_analysis(audio_tensor, emotion_results, save_path)
    
    # Create comparison analysis
    print(f"\n📈 Creating emotion comparison analysis...")
    
    # Compare mean activations across emotions
    comparison_data = {}
    for layer_name in ["layer_2", "layer_6", "layer_10"]:
        comparison_data[layer_name] = {}
        for neuron_name in ["neuron_64", "neuron_128", "neuron_256", "neuron_384"]:
            comparison_data[layer_name][neuron_name] = {}
            for emotion in emotions:
                try:
                    mean_act = all_results[emotion][layer_name][neuron_name]['mean_activation']
                    comparison_data[layer_name][neuron_name][emotion] = mean_act
                except KeyError:
                    comparison_data[layer_name][neuron_name][emotion] = 0.0
    
    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for idx, layer_name in enumerate(["layer_2", "layer_6", "layer_10"]):
        ax = axes[idx]
        
        neuron_names = list(comparison_data[layer_name].keys())
        x = np.arange(len(neuron_names))
        width = 0.2
        
        for i, emotion in enumerate(emotions):
            values = [comparison_data[layer_name][neuron][emotion] for neuron in neuron_names]
            ax.bar(x + i * width, values, width, label=emotion)
        
        ax.set_title(f"{layer_name} - Mean Neuron Activations by Emotion")
        ax.set_xlabel("Neurons")
        ax.set_ylabel("Mean Activation")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(neuron_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    comparison_path = results_dir / "emotion_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Analysis complete! Results saved in: {results_dir}")
    print(f"📊 Comparison plot saved: {comparison_path}")
    
    # Print summary statistics
    print(f"\n📋 Summary Statistics:")
    for emotion in emotions:
        print(f"\n{emotion.upper()}:")
        for layer_name in ["layer_2", "layer_6", "layer_10"]:
            max_neuron = ""
            max_activation = -float('inf')
            for neuron_name in ["neuron_64", "neuron_128", "neuron_256", "neuron_384"]:
                try:
                    activation = all_results[emotion][layer_name][neuron_name]['mean_activation']
                    if activation > max_activation:
                        max_activation = activation
                        max_neuron = neuron_name
                except KeyError:
                    continue
            print(f"  {layer_name}: Highest activation in {max_neuron} ({max_activation:.4f})")
    
    return all_results

if __name__ == "__main__":
    # Run the simplified test
    results = test_basic_wavlm_analysis()