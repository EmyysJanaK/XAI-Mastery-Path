#!/usr/bin/env python3

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from LayerwiseLowLevelAblation import LayerwiseLowLevelAblation

def run_ablation_analysis(ravdess_path, output_dir, emotion_class=2, num_samples=300, batch_size=8, 
                         topk_neurons=20, frame_policy='max_frame', model_name='microsoft/wavlm-base-plus'):
    """
    Run layerwise low-level ablation analysis on WavLM using the RAVDESS dataset.
    
    Args:
        ravdess_path (str): Path to the RAVDESS dataset
        output_dir (str): Directory to save results
        emotion_class (int): Emotion class to target (0=neutral, 1=calm, 2=happy, 3=sad, 
                            4=angry, 5=fearful, 6=disgust, 7=surprised)
        num_samples (int): Number of samples to use from dataset
        batch_size (int): Batch size for processing
        topk_neurons (int): Number of top neurons to ablate per layer
        frame_policy (str): Ablation policy ('max_frame' or 'all_frames')
        model_name (str): HuggingFace model name for WavLM
    
    Returns:
        tuple: Results dictionary and path to visualization figure
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Map numeric emotion class to name for display
    emotion_names = [
        'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
    ]
    emotion_name = emotion_names[emotion_class] if 0 <= emotion_class < len(emotion_names) else f"class_{emotion_class}"
    
    print(f"🎵 Starting WavLM Layer-wise Low-Level Ablation Analysis")
    print(f"====================================================================")
    print(f"Target emotion: {emotion_name} (class {emotion_class})")
    print(f"Model: {model_name}")
    print(f"Dataset: {ravdess_path}")
    print(f"Analysis settings:")
    print(f"  - Number of samples: {num_samples}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Top-k neurons: {topk_neurons}")
    print(f"  - Frame policy: {frame_policy}")
    print(f"====================================================================")
    
    # Initialize the ablation analyzer
    ablator = LayerwiseLowLevelAblation(
        model_name=model_name,
        data_path=ravdess_path,
        save_dir=output_dir,
        max_samples=num_samples,
        batch_size=batch_size,
        random_seed=42
    )
    
    # Run analysis
    try:
        results, fig_path = ablator.run_complete_analysis(
            target_class=emotion_class,
            topk=topk_neurons,
            frame_policy=frame_policy,
            num_batches=num_samples // batch_size + 1  # Ensure we process all samples
        )
        
        print(f"\n✅ Analysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Visualization saved to: {fig_path}")
        
        # Display summary of findings
        print("\n📊 Analysis Summary:")
        print("--------------------------------------------------------------------")
        
        # Extract key information from results
        layers = sorted(list(results['layers'].keys()))
        
        # Find layer with maximum effect on each feature proxy
        rms_changes = [results['layers'][l]['relative_changes']['rms_mean'] for l in layers]
        zcr_changes = [results['layers'][l]['relative_changes']['zcr_mean'] for l in layers]
        sc_changes = [results['layers'][l]['relative_changes']['sc_mean'] for l in layers]
        
        max_rms_layer = layers[np.argmax(rms_changes)]
        max_zcr_layer = layers[np.argmax(zcr_changes)]
        max_sc_layer = layers[np.argmax(sc_changes)]
        
        print(f"Most sensitive layer for RMS Energy proxy: Layer {max_rms_layer}")
        print(f"Most sensitive layer for Zero-Crossing Rate proxy: Layer {max_zcr_layer}")
        print(f"Most sensitive layer for Spectral Centroid proxy: Layer {max_sc_layer}")
        print("--------------------------------------------------------------------")
        
        return results, fig_path
        
    except Exception as e:
        print(f"❌ Error during ablation analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Run WavLM layerwise low-level ablation analysis')
    parser.add_argument('--ravdess_path', type=str, required=True, 
                      help='Path to the RAVDESS dataset')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                      help='Directory to save results')
    parser.add_argument('--emotion_class', type=int, default=2,
                      help='Emotion class to target (0-7, 2=happy)')
    parser.add_argument('--num_samples', type=int, default=300,
                      help='Number of samples to use from dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for processing')
    parser.add_argument('--topk_neurons', type=int, default=20,
                      help='Number of top neurons to ablate per layer')
    parser.add_argument('--frame_policy', type=str, default='max_frame', 
                      choices=['max_frame', 'all_frames'],
                      help='Ablation policy (max_frame or all_frames)')
    parser.add_argument('--model_name', type=str, default='microsoft/wavlm-base-plus',
                      help='HuggingFace model name for WavLM')
    
    args = parser.parse_args()
    
    # Run analysis
    run_ablation_analysis(
        ravdess_path=args.ravdess_path,
        output_dir=args.output_dir,
        emotion_class=args.emotion_class,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        topk_neurons=args.topk_neurons,
        frame_policy=args.frame_policy,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()