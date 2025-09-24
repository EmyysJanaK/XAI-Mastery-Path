#!/usr/bin/env python3

"""
Complete integration script for WavLM Neuron Ablation with RAVDESS dataset
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from wavlm_neuron_ablation import WavLMNeuronAblation
from ravdess_handler import RAVDESSDataHandler

def run_wavlm_neuron_analysis(ravdess_path: str, 
                             emotion: str = "happy",
                             layer_idx: int = 6,
                             neuron_dim: int = 256,
                             output_dir: str = "results"):
    """
    Complete pipeline for WavLM neuron ablation analysis on RAVDESS
    
    Args:
        ravdess_path: Path to RAVDESS dataset
        emotion: Target emotion to analyze
        layer_idx: Which transformer layer to analyze
        neuron_dim: Which hidden dimension (neuron) to focus on
        output_dir: Directory to save results
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🎵 Starting WavLM Neuron Ablation Analysis")
    print("=" * 50)
    
    # Step 1: Initialize components
    print("1. Initializing WavLM model...")
    analyzer = WavLMNeuronAblation()
    
    print("2. Loading RAVDESS dataset...")
    data_handler = RAVDESSDataHandler(ravdess_path)
    data_handler.print_dataset_summary()
    
    # Step 2: Select and load audio sample
    print(f"\n3. Selecting {emotion} sample...")
    try:
        sample_info = data_handler.get_random_sample(emotion=emotion)
        print(f"   Selected: {sample_info['filename']}")
        print(f"   Actor: {sample_info['actor']} ({sample_info['gender']})")
        print(f"   Intensity: {sample_info['intensity']}")
        
        # Load audio
        audio_input = analyzer.preprocess_audio(sample_info['file_path'])
        print(f"   Audio shape: {audio_input.shape}")
        
    except Exception as e:
        print(f"   Error loading sample: {e}")
        return None
    
    # Step 3: Perform neuron ablation analysis
    print(f"\n4. Analyzing neuron {neuron_dim} in layer {layer_idx}...")
    
    try:
        # Analysis 1: Average over sequence positions
        print("   - Analysis 1: Average neuron activation across sequence")
        attributions_avg = analyzer.ablate_neuron(
            audio_input=audio_input,
            layer_idx=layer_idx,
            neuron_selector=neuron_dim,  # Single dimension, averaged over sequence
            baseline_value=0.0,
            perturbations_per_eval=8
        )
        
        # Analysis 2: Specific sequence position
        seq_pos = audio_input.shape[-1] // 2  # Middle of sequence
        print(f"   - Analysis 2: Specific neuron at sequence position {seq_pos}")
        attributions_specific = analyzer.ablate_neuron(
            audio_input=audio_input,
            layer_idx=layer_idx,
            neuron_selector=(seq_pos, neuron_dim),  # Specific position
            baseline_value=0.0,
            perturbations_per_eval=8
        )
        
        # Step 4: Visualize and save results
        print("\n5. Generating visualizations...")
        
        # Plot 1: Average neuron analysis
        plt.figure(figsize=(15, 10))
        
        # Original audio waveform
        plt.subplot(3, 1, 1)
        audio_np = audio_input.squeeze().detach().cpu().numpy()
        time_axis = np.linspace(0, len(audio_np) / 16000, len(audio_np))
        plt.plot(time_axis, audio_np)
        plt.title(f"Original Audio: {sample_info['filename']} ({emotion})")
        plt.ylabel("Amplitude")
        
        # Attribution - averaged neuron
        plt.subplot(3, 1, 2)
        attr_avg_np = attributions_avg.squeeze().detach().cpu().numpy()
        time_axis_attr = np.linspace(0, len(attr_avg_np) / 16000, len(attr_avg_np))
        plt.plot(time_axis_attr, attr_avg_np, color='red', linewidth=2)
        plt.title(f"Feature Attributions - Layer {layer_idx}, Neuron {neuron_dim} (Averaged)")
        plt.ylabel("Attribution Score")
        
        # Attribution - specific position
        plt.subplot(3, 1, 3)
        attr_spec_np = attributions_specific.squeeze().detach().cpu().numpy()
        plt.plot(time_axis_attr, attr_spec_np, color='blue', linewidth=2)
        plt.title(f"Feature Attributions - Layer {layer_idx}, Neuron {neuron_dim} (Position {seq_pos})")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Attribution Score")
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / f"wavlm_ablation_{emotion}_layer{layer_idx}_neuron{neuron_dim}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   Saved plot: {plot_path}")
        
        # Step 5: Statistical analysis
        print("\n6. Statistical Analysis:")
        print(f"   Average attribution magnitude: {torch.abs(attributions_avg).mean().item():.6f}")
        print(f"   Attribution range: [{attributions_avg.min().item():.6f}, {attributions_avg.max().item():.6f}]")
        print(f"   Most important time region: {torch.argmax(torch.abs(attributions_avg)).item() / 16000:.3f}s")
        
        # Save results
        results = {
            'sample_info': sample_info,
            'layer_idx': layer_idx,
            'neuron_dim': neuron_dim,
            'attributions_avg': attributions_avg.detach().cpu(),
            'attributions_specific': attributions_specific.detach().cpu(),
            'audio_input': audio_input.detach().cpu()
        }
        
        results_path = output_path / f"ablation_results_{emotion}_layer{layer_idx}_neuron{neuron_dim}.pt"
        torch.save(results, results_path)
        print(f"   Saved results: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"   Error during analysis: {e}")
        return None

def compare_emotions_neuron_response(ravdess_path: str, 
                                   emotions: list = ["happy", "sad", "angry"],
                                   layer_idx: int = 6,
                                   neuron_dim: int = 256,
                                   output_dir: str = "results"):
    """
    Compare how the same neuron responds to different emotions
    
    Args:
        ravdess_path: Path to RAVDESS dataset
        emotions: List of emotions to compare
        layer_idx: Which transformer layer to analyze
        neuron_dim: Which neuron to focus on
        output_dir: Directory to save results
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🔬 Comparing neuron {neuron_dim} response across emotions: {emotions}")
    print("=" * 60)
    
    analyzer = WavLMNeuronAblation()
    data_handler = RAVDESSDataHandler(ravdess_path)
    
    results = {}
    
    plt.figure(figsize=(15, len(emotions) * 4))
    
    for i, emotion in enumerate(emotions):
        print(f"\nAnalyzing {emotion}...")
        
        try:
            # Get sample
            sample_info = data_handler.get_random_sample(emotion=emotion)
            audio_input = analyzer.preprocess_audio(sample_info['file_path'])
            
            # Ablation analysis
            attributions = analyzer.ablate_neuron(
                audio_input=audio_input,
                layer_idx=layer_idx,
                neuron_selector=neuron_dim,
                perturbations_per_eval=8
            )
            
            results[emotion] = {
                'attributions': attributions,
                'sample_info': sample_info,
                'audio_input': audio_input
            }
            
            # Plot
            plt.subplot(len(emotions), 1, i + 1)
            attr_np = attributions.squeeze().detach().cpu().numpy()
            time_axis = np.linspace(0, len(attr_np) / 16000, len(attr_np))
            plt.plot(time_axis, attr_np, linewidth=2, label=f"{emotion} (Actor {sample_info['actor']})")
            plt.title(f"Neuron {neuron_dim} Response - {emotion.capitalize()}")
            plt.ylabel("Attribution")
            if i == len(emotions) - 1:
                plt.xlabel("Time (seconds)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            print(f"   Mean attribution: {torch.abs(attributions).mean().item():.6f}")
            
        except Exception as e:
            print(f"   Error processing {emotion}: {e}")
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = output_path / f"emotion_comparison_layer{layer_idx}_neuron{neuron_dim}.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    torch.save(results, output_path / f"emotion_comparison_results_layer{layer_idx}_neuron{neuron_dim}.pt")
    
    print(f"\n✅ Comparison complete! Results saved to {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="WavLM Neuron Ablation on RAVDESS")
    parser.add_argument("--ravdess_path", type=str, required=True, 
                       help="Path to RAVDESS dataset")
    parser.add_argument("--emotion", type=str, default="happy",
                       help="Target emotion to analyze")
    parser.add_argument("--layer", type=int, default=6,
                       help="Transformer layer to analyze")
    parser.add_argument("--neuron", type=int, default=256,
                       help="Neuron dimension to analyze")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--compare", action="store_true",
                       help="Run emotion comparison analysis")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_emotions_neuron_response(
            ravdess_path=args.ravdess_path,
            emotions=["happy", "sad", "angry", "fearful"],
            layer_idx=args.layer,
            neuron_dim=args.neuron,
            output_dir=args.output
        )
    else:
        run_wavlm_neuron_analysis(
            ravdess_path=args.ravdess_path,
            emotion=args.emotion,
            layer_idx=args.layer,
            neuron_dim=args.neuron,
            output_dir=args.output
        )

if __name__ == "__main__":
    # Example usage without command line args
    # Replace with your actual RAVDESS path
    RAVDESS_PATH = "/path/to/your/ravdess/audio_speech_actors_01-24/"
    
    print("🚀 Example: Single emotion analysis")
    results = run_wavlm_neuron_analysis(
        ravdess_path=RAVDESS_PATH,
        emotion="happy",
        layer_idx=8,
        neuron_dim=384,
        output_dir="wavlm_results"
    )
    
    if results:
        print("\n🚀 Example: Multi-emotion comparison")
        comparison = compare_emotions_neuron_response(
            ravdess_path=RAVDESS_PATH,
            emotions=["happy", "sad", "angry"],
            layer_idx=8,
            neuron_dim=384,
            output_dir="wavlm_results"
        )