"""
Example usage of the neuron ablation framework.
"""

import os
import torch
import logging
import argparse
from transformers import AutoModel
from datetime import datetime

from neuron_ablation.core import NeuronAblator
from neuron_ablation.visualization import ResultsVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('example')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run neuron ablation analysis')
    parser.add_argument('--model_name', type=str, default='microsoft/wavlm-base-plus',
                        help='Model name or path (default: microsoft/wavlm-base-plus)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset for ablation')
    parser.add_argument('--save_dir', type=str, default='ablation_results',
                        help='Directory to save results (default: ablation_results)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for ablation (default: 4)')
    parser.add_argument('--topk', type=int, default=100,
                        help='Top-K neurons to ablate (default: 100)')
    parser.add_argument('--target_class', type=int, default=0,
                        help='Target class index (default: 0)')
    parser.add_argument('--ablate_mode', type=str, default='zero',
                        choices=['zero', 'mean', 'noise', 'invert'],
                        help='Ablation mode (default: zero)')
    parser.add_argument('--attribution_method', type=str, default='integrated_gradients',
                        choices=['integrated_gradients', 'direct_gradient', 'ablation', 'activation'],
                        help='Method for attribution (default: integrated_gradients)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to use (default: 10, -1 for all)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available, else cpu)')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.save_dir,
        f"{args.attribution_method}_{args.ablate_mode}_class{args.target_class}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Results will be saved to: {save_dir}")
    
    try:
        # Load the model
        logger.info(f"Loading model: {args.model_name}")
        model = AutoModel.from_pretrained(args.model_name)
        model.to(args.device)
        model.eval()
        
        # Define layer groups for structured analysis
        layer_groups = {
            'input': [0],
            'encoder': list(range(1, len(model.encoder.layers))),
            'output': [len(model.encoder.layers)]
        }
        
        # Initialize neuron ablator
        ablator = NeuronAblator(
            model=model,
            save_dir=save_dir,
            layer_groups=layer_groups,
            device=args.device
        )
        
        # Load dataset
        # Note: This is a placeholder. You'll need to implement data loading for your specific dataset
        logger.info(f"Loading dataset from: {args.data_path}")
        # dataset = load_your_dataset(args.data_path)
        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # For demonstration purposes only (replace with actual data loading)
        logger.warning("Using dummy data for demonstration. Replace with actual data loading.")
        dummy_inputs = torch.randn(args.batch_size, 16000, device=args.device)  # Dummy audio inputs
        dummy_labels = torch.randint(0, 8, (args.batch_size,), device=args.device)  # Dummy emotion labels
        
        # Run ablation
        logger.info(f"Running ablation with {args.attribution_method} attribution")
        results = ablator.run_layerwise_ablation(
            inputs=dummy_inputs,
            labels=dummy_labels,
            topk=args.topk,
            ablate_mode=args.ablate_mode,
            target_class=args.target_class,
            attribution_method=args.attribution_method
        )
        
        # Visualize results
        logger.info("Generating visualizations")
        visualizer = ResultsVisualizer(ablator)
        visualizer.generate_all_plots(results)
        
        logger.info("Ablation analysis completed successfully")
        
    except Exception as e:
        logger.exception(f"Error during ablation analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())