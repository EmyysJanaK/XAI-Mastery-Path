#!/usr/bin/env python3
"""
gradientshap_demo.py: Demo script for using GradientSHAP with WavLM and librosa.

This script provides a complete demonstration of how to use the GradientSHAP
implementation to explain WavLM model predictions on speech data.
"""

import os
import argparse
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from explainability.LIME_SHAP.ShapeExplainer import GradientSHAP, WavLMGradientSHAPDemo

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GradientSHAP Demo for WavLM')
    
    parser.add_argument('--audio_file', type=str, 
                        help='Path to audio file for explanation')
    
    parser.add_argument('--data_dir', type=str, 
                        help='Directory containing audio files for batch processing')
    
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save explanation visualizations')
    
    parser.add_argument('--target_class', type=int, default=None,
                        help='Target class for explanation (0-7 for emotions)')
    
    parser.add_argument('--model', type=str, default='microsoft/wavlm-base-plus',
                        help='WavLM model name or path')
    
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of reference samples for GradientSHAP')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing')
    
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top frames to highlight in visualization')
    
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def explain_single_file(args):
    """Generate explanation for a single audio file."""
    print(f"Explaining audio file: {args.audio_file}")
    
    # Check if file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found - {args.audio_file}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create output filename
    basename = os.path.basename(args.audio_file)
    output_path = os.path.join(args.output_dir, f"explanation_{basename}.png")
    
    # Initialize explainer
    explainer = GradientSHAP(
        model_name=args.model,
        device=args.device,
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
    
    # Visualize explanations
    try:
        explanation = explainer.visualize_explanations(
            audio_path=args.audio_file,
            target_class=args.target_class,
            save_path=output_path,
            top_k_frames=args.top_k
        )
        
        print(f"Explanation generated successfully!")
        print(f"Visualization saved to: {output_path}")
        
        # Print prediction
        pred_class = explanation['prediction']['class']
        pred_prob = explanation['prediction']['probability']
        print(f"Prediction: Class {pred_class} with probability {pred_prob:.4f}")
        
    except Exception as e:
        print(f"Error generating explanation: {e}")
        import traceback
        traceback.print_exc()


def process_directory(args):
    """Process all audio files in a directory."""
    print(f"Processing audio files from directory: {args.data_dir}")
    
    # Check if directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory not found - {args.data_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the demo class
    demo = WavLMGradientSHAPDemo(
        model_name=args.model,
        data_path=args.data_dir,
        num_classes=8,  # For emotion recognition
        device=args.device,
        batch_size=args.batch_size,
        max_samples=10  # Process a maximum of 10 samples for demo
    )
    
    # Run the demo
    try:
        results = demo.run_demo(
            target_class=args.target_class,
            save_dir=args.output_dir
        )
        
        print(f"\nProcessed {len(results)} audio files successfully!")
        print(f"Visualizations saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error processing directory: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the GradientSHAP demo."""
    args = parse_arguments()
    
    print("\n" + "=" * 50)
    print("GradientSHAP Demo for WavLM Speech Models")
    print("=" * 50)
    
    # Determine device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {args.device}")
    print(f"Model: {args.model}")
    print(f"Reference samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 50 + "\n")
    
    # Either process single file or directory
    if args.audio_file:
        explain_single_file(args)
    elif args.data_dir:
        process_directory(args)
    else:
        print("Error: Please specify either --audio_file or --data_dir")
        return
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()