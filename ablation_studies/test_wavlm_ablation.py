#!/usr/bin/env python3

"""
Test script for WavLM Neuron Ablation with synthetic audio data
This tests the implementation without requiring the RAVDESS dataset
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom classes
from wavlm_neuron_ablation import WavLMNeuronAblation

def create_synthetic_audio(duration=3.0, sample_rate=16000, emotion_type="happy"):
    """
    Create synthetic audio data for testing
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        emotion_type: Type of emotion to simulate
        
    Returns:
        Synthetic audio tensor
    """
    print(f"Creating synthetic {emotion_type} audio ({duration}s, {sample_rate}Hz)")
    
    # Generate synthetic speech-like signal
    t = torch.linspace(0, duration, int(duration * sample_rate))
    
    if emotion_type == "happy":
        # Higher frequency components, more variation
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
        # Neutral - simple sine wave
        signal = torch.sin(2 * np.pi * 120 * t) + 0.1 * torch.randn(len(t))
    
    # Normalize
    signal = signal / torch.max(torch.abs(signal)) * 0.8
    
    return signal.unsqueeze(0)  # Add batch dimension

def test_wavlm_loading():
    """Test if WavLM model loads correctly"""
    print("\n" + "="*50)
    print("TEST 1: WavLM Model Loading")
    print("="*50)
    
    try:
        analyzer = WavLMNeuronAblation()
        print("✅ WavLM model loaded successfully!")
        print(f"   - Device: {analyzer.device}")
        print(f"   - Number of layers: {analyzer.num_layers}")
        return analyzer
    except Exception as e:
        print(f"❌ Error loading WavLM: {e}")
        return None

def test_audio_preprocessing(analyzer):
    """Test audio preprocessing"""
    print("\n" + "="*50)
    print("TEST 2: Audio Preprocessing")
    print("="*50)
    
    try:
        # Create synthetic audio
        synthetic_audio = create_synthetic_audio(duration=2.0)
        
        # Save temporary audio file
        temp_path = "temp_synthetic_audio.wav"
        torchaudio.save(temp_path, synthetic_audio, 16000)
        
        # Test preprocessing
        processed_audio = analyzer.preprocess_audio(temp_path)
        
        print(f"✅ Audio preprocessing successful!")
        print(f"   - Original shape: {synthetic_audio.shape}")
        print(f"   - Processed shape: {processed_audio.shape}")
        
        # Cleanup
        Path(temp_path).unlink()
        
        return processed_audio
    except Exception as e:
        print(f"❌ Error in audio preprocessing: {e}")
        return None

def test_neuron_selection(analyzer, audio_input):
    """Test neuron selection mechanisms"""
    print("\n" + "="*50)
    print("TEST 3: Neuron Selection")
    print("="*50)
    
    try:
        layer_idx = 3  # Test with layer 3
        
        # Test 1: Single neuron dimension
        print(f"Testing neuron selection in layer {layer_idx}")
        neuron_forward_func = analyzer.create_neuron_forward_func(
            layer_idx=layer_idx,
            neuron_selector=128,  # Hidden dimension 128
            attribute_to_input=False
        )
        
        result = neuron_forward_func(audio_input)
        print(f"✅ Single neuron selection works!")
        print(f"   - Result shape: {result.shape}")
        
        # Test 2: Specific position
        neuron_forward_func_pos = analyzer.create_neuron_forward_func(
            layer_idx=layer_idx,
            neuron_selector=(50, 128),  # Position 50, dimension 128
            attribute_to_input=False
        )
        
        result_pos = neuron_forward_func_pos(audio_input)
        print(f"✅ Position-specific neuron selection works!")
        print(f"   - Result shape: {result_pos.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error in neuron selection: {e}")
        return False

def test_ablation_analysis(analyzer, audio_input):
    """Test the actual ablation analysis"""
    print("\n" + "="*50)
    print("TEST 4: Ablation Analysis")
    print("="*50)
    
    try:
        layer_idx = 2
        neuron_dim = 64
        
        print(f"Running ablation on layer {layer_idx}, neuron {neuron_dim}")
        print("This might take a moment...")
        
        # Run ablation with fewer perturbations for faster testing
        attributions = analyzer.ablate_neuron(
            audio_input=audio_input,
            layer_idx=layer_idx,
            neuron_selector=neuron_dim,
            baseline_value=0.0,
            perturbations_per_eval=4  # Smaller for faster testing
        )
        
        print(f"✅ Ablation analysis successful!")
        print(f"   - Attribution shape: {attributions.shape}")
        print(f"   - Attribution range: [{attributions.min().item():.4f}, {attributions.max().item():.4f}]")
        print(f"   - Mean absolute attribution: {torch.abs(attributions).mean().item():.4f}")
        
        return attributions
    except Exception as e:
        print(f"❌ Error in ablation analysis: {e}")
        return None

def test_visualization(analyzer, attributions, audio_input):
    """Test visualization functionality"""
    print("\n" + "="*50)
    print("TEST 5: Visualization")
    print("="*50)
    
    try:
        # Create results directory
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        # Test visualization
        save_path = results_dir / "test_ablation_visualization.png"
        analyzer.visualize_attributions(
            attributions=attributions,
            audio_input=audio_input,
            save_path=str(save_path)
        )
        
        print(f"✅ Visualization successful!")
        print(f"   - Plot saved: {save_path}")
        
        return True
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
        return False

def test_multiple_emotions():
    """Test with multiple synthetic emotion types"""
    print("\n" + "="*50)
    print("TEST 6: Multi-Emotion Analysis")
    print("="*50)
    
    try:
        analyzer = WavLMNeuronAblation()
        emotions = ["happy", "sad", "angry", "neutral"]
        results = {}
        
        layer_idx = 4
        neuron_dim = 128
        
        print(f"Analyzing neuron {neuron_dim} in layer {layer_idx} across emotions:")
        
        plt.figure(figsize=(15, 10))
        
        for i, emotion in enumerate(emotions):
            print(f"  Processing {emotion}...")
            
            # Create synthetic audio for this emotion
            synthetic_audio = create_synthetic_audio(duration=1.5, emotion_type=emotion)
            
            # Save and process
            temp_path = f"temp_{emotion}_audio.wav"
            torchaudio.save(temp_path, synthetic_audio, 16000)
            audio_input = analyzer.preprocess_audio(temp_path)
            
            # Run ablation
            attributions = analyzer.ablate_neuron(
                audio_input=audio_input,
                layer_idx=layer_idx,
                neuron_selector=neuron_dim,
                perturbations_per_eval=2  # Fast testing
            )
            
            results[emotion] = {
                'attributions': attributions,
                'mean_abs_attr': torch.abs(attributions).mean().item()
            }
            
            # Plot
            plt.subplot(2, 2, i + 1)
            attr_np = attributions.squeeze().detach().cpu().numpy()
            time_axis = np.linspace(0, len(attr_np) / 16000, len(attr_np))
            plt.plot(time_axis, attr_np, linewidth=2)
            plt.title(f"{emotion.capitalize()} - Neuron {neuron_dim}")
            plt.xlabel("Time (s)")
            plt.ylabel("Attribution")
            plt.grid(True, alpha=0.3)
            
            # Cleanup
            Path(temp_path).unlink()
        
        plt.tight_layout()
        
        # Save comparison plot
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        comparison_path = results_dir / "emotion_comparison_test.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Multi-emotion analysis successful!")
        print(f"   - Comparison plot saved: {comparison_path}")
        
        # Print statistics
        print("\nEmotion Response Statistics:")
        for emotion, data in results.items():
            print(f"   {emotion:>8}: Mean |attribution| = {data['mean_abs_attr']:.4f}")
        
        return results
    except Exception as e:
        print(f"❌ Error in multi-emotion analysis: {e}")
        return None

def run_full_test_suite():
    """Run the complete test suite"""
    print("🚀 Starting WavLM Neuron Ablation Test Suite")
    print("="*70)
    
    # Test 1: Model Loading
    analyzer = test_wavlm_loading()
    if analyzer is None:
        print("\n❌ Test suite failed at model loading")
        return
    
    # Test 2: Audio Preprocessing
    audio_input = test_audio_preprocessing(analyzer)
    if audio_input is None:
        print("\n❌ Test suite failed at audio preprocessing")
        return
    
    # Test 3: Neuron Selection
    if not test_neuron_selection(analyzer, audio_input):
        print("\n❌ Test suite failed at neuron selection")
        return
    
    # Test 4: Ablation Analysis
    attributions = test_ablation_analysis(analyzer, audio_input)
    if attributions is None:
        print("\n❌ Test suite failed at ablation analysis")
        return
    
    # Test 5: Visualization
    if not test_visualization(analyzer, attributions, audio_input):
        print("\n❌ Test suite failed at visualization")
        return
    
    # Test 6: Multi-Emotion Analysis
    emotion_results = test_multiple_emotions()
    if emotion_results is None:
        print("\n❌ Test suite failed at multi-emotion analysis")
        return
    
    # All tests passed!
    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("="*70)
    print("\nYour WavLM Neuron Ablation implementation is working correctly!")
    print("\nNext steps:")
    print("1. Download the RAVDESS dataset")
    print("2. Update paths in wavlm_ravdess_integration.py")
    print("3. Run real emotion analysis!")
    print("\nTest results saved in: test_results/")

if __name__ == "__main__":
    run_full_test_suite()