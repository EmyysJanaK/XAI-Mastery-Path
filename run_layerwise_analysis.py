#!/usr/bin/env python3

import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from LayerwiseLowLevelAblation import LayerwiseLowLevelAblation

def main():
    parser = argparse.ArgumentParser(description='Run WavLM sign-flip ablation analysis with low-level feature proxies')
    
    parser.add_argument('--ravdess_path', type=str, required=True, 
                       help='Path to the RAVDESS dataset')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                       help='Directory to save results')
    parser.add_argument('--emotion', type=str, default='happy',
                       choices=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
                       help='Target emotion to analyze')
    parser.add_argument('--samples', type=int, default=200,
                       help='Number of audio samples to analyze')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing')
    parser.add_argument('--topk', type=int, default=50,
                       help='Number of top neurons to ablate')
    parser.add_argument('--model', type=str, default='microsoft/wavlm-base-plus',
                       help='HuggingFace model name')
    parser.add_argument('--frame_policy', type=str, default='max_frame',
                       choices=['max_frame', 'all_frames'],
                       help='Ablation frame policy')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu, defaults to cuda if available)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Map emotion name to class index
    emotion_classes = {
        'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
        'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
    }
    target_class = emotion_classes.get(args.emotion, 2)  # Default to 'happy' if not found
    
    print(f"🎵 WavLM Layer-wise Low-Level Ablation Analysis")
    print(f"====================================================")
    print(f"Target emotion: {args.emotion} (class {target_class})")
    print(f"Model: {args.model}")
    print(f"Dataset path: {args.ravdess_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Analysis settings:")
    print(f"  - Samples: {args.samples}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Top-k neurons: {args.topk}")
    print(f"  - Frame policy: {args.frame_policy}")
    print(f"====================================================\n")
    
    try:
        # Initialize the ablation analyzer
        ablator = LayerwiseLowLevelAblation(
            model_name=args.model,
            data_path=args.ravdess_path,
            save_dir=args.output_dir,
            max_samples=args.samples,
            batch_size=args.batch_size,
            device=args.device,
            random_seed=42
        )
        
        # Run the analysis
        results, fig_path = ablator.run_complete_analysis(
            target_class=target_class,
            topk=args.topk,
            frame_policy=args.frame_policy
        )
        
        print(f"\n✅ Analysis complete!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Visualization saved to: {fig_path}")
        
        # Display summary information
        layers = sorted(list(results['layers'].keys()))
        
        # Find the layer with the maximum effect for each proxy
        rms_changes = [results['layers'][l]['relative_changes']['rms_mean'] for l in layers]
        zcr_changes = [results['layers'][l]['relative_changes']['zcr_mean'] for l in layers]
        sc_changes = [results['layers'][l]['relative_changes']['sc_mean'] for l in layers]
        
        max_rms_layer = layers[max(range(len(rms_changes)), key=lambda i: rms_changes[i])]
        max_zcr_layer = layers[max(range(len(zcr_changes)), key=lambda i: zcr_changes[i])]
        max_sc_layer = layers[max(range(len(sc_changes)), key=lambda i: sc_changes[i])]
        
        print("\n📊 Analysis Summary:")
        print("----------------------------------------------------")
        print(f"Most sensitive layer for RMS Energy: Layer {max_rms_layer}")
        print(f"Most sensitive layer for Zero-Crossing Rate: Layer {max_zcr_layer}")
        print(f"Most sensitive layer for Spectral Centroid: Layer {max_sc_layer}")
        print("----------------------------------------------------")
        
        # Print the top 5 most important neurons for one of these layers
        sample_layer = max_rms_layer  # Use the most sensitive layer for RMS
        if sample_layer in results['important_neurons']:
            top_neurons = results['important_neurons'][sample_layer][:5]
            print(f"\nTop 5 important neurons in layer {sample_layer}: {top_neurons}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()