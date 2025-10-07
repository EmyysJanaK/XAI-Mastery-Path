"""
Visualization utilities for ablation results.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger('ResultsVisualizer')

class ResultsVisualizer:
    """
    Generate visualizations from ablation results.
    
    This class provides optimized plotting functions that generate:
    1. Feature changes across layers
    2. Low-level feature analysis
    3. Temporal spread heatmaps
    4. Upstream norms visualization
    5. Emotion probability changes
    """
    
    def __init__(self, ablator):
        """
        Initialize the ResultsVisualizer.
        
        Args:
            ablator: Parent NeuronAblator instance
        """
        self.ablator = ablator
        self.save_dir = ablator.save_dir
        self.layer_groups = ablator.layer_groups
        self.emotion_classes = ablator.emotion_classes
        
    def generate_all_plots(self, results):
        """
        Generate all visualization plots for the ablation results.
        
        Args:
            results: Results dictionary from run_layerwise_ablation
        """
        # Check for required data
        if 'ablated' not in results or not results['ablated']:
            logger.warning("No ablation results available for visualization")
            return
            
        # Create a summary file
        self._create_summary(results)
            
        try:
            # Plot 1: Feature changes across layers
            self.plot_feature_changes(results)
        except Exception as e:
            logger.error(f"Error plotting feature changes: {e}")
        
        try:
            # Plot 2: Specific low-level feature changes
            self.plot_low_level_features(results)
        except Exception as e:
            logger.error(f"Error plotting low-level feature changes: {e}")
        
        try:
            # Plot 3: Temporal spread heatmap
            self.plot_temporal_spread(results)
        except Exception as e:
            logger.error(f"Error plotting temporal spread: {e}")
        
        try:
            # Plot 4: Upstream norm changes
            self.plot_upstream_norms(results)
        except Exception as e:
            logger.error(f"Error plotting upstream norms: {e}")
        
        try:
            # Plot 5: Emotion probability changes
            self.plot_emotion_prob_changes(results)
        except Exception as e:
            logger.error(f"Error plotting emotion probability changes: {e}")
    
    def _create_summary(self, results):
        """Create a text summary of the ablation results"""
        # Extract configuration
        config = results['configs']
        topk = config.get('topk', 0)
        ablate_mode = config.get('ablate_mode', 'unknown')
        target_class = config.get('target_class', 0)
        attribution_method = config.get('attribution_method', 'unknown')
        
        # Create a summary file
        summary_path = os.path.join(self.save_dir, f"ablation_summary_class{target_class}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Neuron Ablation Summary\n")
            f.write(f"=====================\n\n")
            f.write(f"Configuration:\n")
            f.write(f"- Target class: {target_class} ({self.emotion_classes[target_class] if target_class < len(self.emotion_classes) else 'Unknown'})\n")
            f.write(f"- Top-K neurons: {topk}\n")
            f.write(f"- Ablation mode: {ablate_mode}\n")
            f.write(f"- Frame policy: {config.get('frame_policy', 'unknown')}\n")
            f.write(f"- Upstream inversion: {config.get('invert', False)}\n")
            f.write(f"- Attribution method: {attribution_method}\n\n")
            
            # Write layer group information
            f.write(f"Layer groups:\n")
            for level, layers in self.layer_groups.items():
                f.write(f"- {level}: {layers}\n")
            f.write("\n")
            
            # Write important neurons information if available
            if 'important_neurons' in results['baseline']:
                f.write(f"Important neurons:\n")
                for layer_idx, neurons in results['baseline']['important_neurons'].items():
                    f.write(f"- Layer {layer_idx}: {neurons[:10]}...\n")
                f.write("\n")
            
            # Write metrics for each layer
            f.write(f"Layer-wise metrics:\n")
            for layer_idx in sorted(results['ablated'].keys()):
                layer_result = results['ablated'][layer_idx]
                
                f.write(f"- Layer {layer_idx}:\n")
                
                # Check if kl_divergence exists
                if 'kl_divergence' in layer_result:
                    try:
                        kl_div = layer_result['kl_divergence'].mean().item() 
                        f.write(f"  - KL divergence: {kl_div:.4f}\n")
                    except (AttributeError, TypeError):
                        f.write(f"  - KL divergence: N/A\n")
                else:
                    f.write(f"  - KL divergence: N/A\n")
                
                # Check if upstream_norm exists
                if 'upstream_norm' in layer_result:
                    try:
                        upstream_norm = layer_result['upstream_norm']
                        f.write(f"  - Upstream norm: {upstream_norm:.4f}\n")
                    except (AttributeError, TypeError):
                        f.write(f"  - Upstream norm: N/A\n")
                else:
                    f.write(f"  - Upstream norm: N/A\n")
                
                # Get level for this layer
                level = None
                for lev, layers in self.layer_groups.items():
                    if layer_idx in layers:
                        level = lev
                        break
                
                if level:
                    f.write(f"  - Level: {level}\n")
                f.write("\n")
        
        logger.info(f"Summary created at {summary_path}")
    
    def plot_feature_changes(self, results):
        """
        Plot how features change with ablation across layers
        """
        # Set up figure
        plt.figure(figsize=(15, 10))
        
        # Get layers in order
        layers = sorted(results['ablated'].keys())
        
        # Track changes for each feature across layers
        feature_changes = {
            'rms_energy': [],
            'amplitude_envelope': [],
            'zero_crossing_rate': [],
            'spectral_centroid': [],
            'pitch_f0': [],
            'f0_delta': [],
            'speaker_similarity': []
        }
        
        # Collect changes for each layer
        for layer_idx in layers:
            layer_result = results['ablated'][layer_idx]
            
            # Skip if feature_deltas is not present
            if 'feature_deltas' not in layer_result:
                # Add dummy values to maintain array length
                for feat_name in feature_changes.keys():
                    feature_changes[feat_name].append((0.0, 0.0))
                continue
            
            # Process each feature
            for feat_name in feature_changes.keys():
                try:
                    if feat_name not in layer_result['feature_deltas']:
                        feature_changes[feat_name].append((0.0, 0.0))
                        continue
                        
                    if feat_name == 'speaker_similarity':
                        # Special handling for cosine similarity
                        changes = layer_result['feature_deltas'][feat_name]
                        # Transform from similarity [0,1] to distance [0,2]
                        changes = 1.0 - changes
                    else:
                        # For other features, use the absolute mean of changes
                        changes = layer_result['feature_deltas'][feat_name]
                        changes = torch.abs(changes).mean(dim=-1)  # Average across frames
                    
                    # Store mean and std across batch
                    mean_change = changes.mean().item()
                    std_change = changes.std().item()
                    feature_changes[feat_name].append((mean_change, std_change))
                except Exception as e:
                    logger.error(f"Error processing feature {feat_name} for layer {layer_idx}: {e}")
                    feature_changes[feat_name].append((0.0, 0.0))
        
        # Create x-axis for layers
        x = np.array(layers)
        
        # Plot each feature
        for i, (feat_name, changes) in enumerate(feature_changes.items()):
            means = np.array([c[0] for c in changes])
            stds = np.array([c[1] for c in changes])
            
            plt.subplot(2, 4, i+1)
            plt.plot(x, means, 'o-', label=feat_name)
            plt.fill_between(x, means-stds, means+stds, alpha=0.3)
            
            # Mark layer groups
            for level, layer_indices in self.layer_groups.items():
                for idx in layer_indices:
                    if idx in x:
                        plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
            
            # Add labels
            plt.title(f"{feat_name.replace('_', ' ').title()} Changes")
            plt.xlabel("Layer Index")
            plt.ylabel("Absolute Mean Change")
            plt.grid(True, alpha=0.3)
        
        # Plot emotion probability changes
        plt.subplot(2, 4, 8)
        emotion_changes = []
        
        for layer_idx in layers:
            layer_result = results['ablated'][layer_idx]
            
            # Get emotion probability deltas
            changes = layer_result['feature_deltas'].get('emotion_probs_delta', None)
            
            if changes is not None:
                # Average across batch
                mean_changes = changes.mean(dim=0)
                emotion_changes.append(mean_changes.cpu().numpy())
        
        if emotion_changes:
            emotion_changes = np.array(emotion_changes)
            plt.imshow(emotion_changes, aspect='auto', cmap='coolwarm')
            plt.colorbar(label='Mean Probability Change')
            plt.xlabel("Emotion Class")
            plt.ylabel("Layer Index")
            plt.title("Emotion Probability Changes")
            
            # Label x-axis with emotion names
            if len(self.emotion_classes) > 0:
                plt.xticks(
                    range(len(self.emotion_classes)),
                    self.emotion_classes,
                    rotation=45
                )
                
            # Label y-axis with layer indices
            plt.yticks(range(len(layers)), layers)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "feature_changes.png"))
        plt.close()
        
        logger.info(f"Feature changes plot saved to {os.path.join(self.save_dir, 'feature_changes.png')}")

    def plot_low_level_features(self, results):
        """
        Plot specific low-level feature changes (energy, amplitude envelope, and zero-crossing rate)
        """
        # Set up figure
        plt.figure(figsize=(20, 14))
        
        # Get layers in order
        layers = sorted(results['ablated'].keys())
        
        # List of low-level features to plot
        low_level_features = ['rms_energy', 'amplitude_envelope', 'zero_crossing_rate']
        feature_names_display = {
            'rms_energy': 'Energy',
            'amplitude_envelope': 'Amplitude Envelope',
            'zero_crossing_rate': 'Zero Crossing Rate'
        }
        feature_colors = {
            'rms_energy': 'tab:blue',
            'amplitude_envelope': 'tab:orange',
            'zero_crossing_rate': 'tab:green'
        }
        
        # Track changes for each feature across layers
        feature_changes = {feat: [] for feat in low_level_features}
        raw_feature_changes = {feat: [] for feat in low_level_features}
        
        # Collect changes for each layer
        for layer_idx in layers:
            layer_result = results['ablated'][layer_idx]
            
            # Process each feature
            for feat_name in low_level_features:
                try:
                    if 'feature_deltas' not in layer_result or feat_name not in layer_result['feature_deltas']:
                        feature_changes[feat_name].append((0.0, 0.0))
                        raw_feature_changes[feat_name].append(None)
                        continue
                    
                    # Get raw changes
                    changes = layer_result['feature_deltas'][feat_name]
                    raw_feature_changes[feat_name].append(changes.cpu().numpy() if hasattr(changes, 'cpu') else changes)
                    
                    # Use the absolute mean of changes
                    changes_abs = torch.abs(changes).mean(dim=-1)  # Average across frames
                    
                    # Store mean and std across batch
                    mean_change = changes_abs.mean().item()
                    std_change = changes_abs.std().item()
                    feature_changes[feat_name].append((mean_change, std_change))
                except Exception as e:
                    logger.error(f"Error processing feature {feat_name} for layer {layer_idx}: {e}")
                    feature_changes[feat_name].append((0.0, 0.0))
                    raw_feature_changes[feat_name].append(None)
        
        # PLOT 1: Combined line plot of all features
        plt.subplot(2, 1, 1)
        for feat_name in low_level_features:
            if not feature_changes[feat_name]:
                continue
                
            means = np.array([c[0] for c in feature_changes[feat_name]])
            stds = np.array([c[1] for c in feature_changes[feat_name]])
            
            plt.plot(layers, means, 'o-', label=feature_names_display[feat_name], 
                    linewidth=2, color=feature_colors[feat_name])
            plt.fill_between(layers, means-stds, means+stds, alpha=0.2, color=feature_colors[feat_name])
        
        # Mark layer groups with colored bands
        colors = ['lightblue', 'lightgreen', 'lightyellow']
        for i, (level, layer_indices) in enumerate(self.layer_groups.items()):
            # Find min and max indices for each group
            valid_indices = [idx for idx in layer_indices if idx in layers]
            if valid_indices:
                min_idx = min(valid_indices)
                max_idx = max(valid_indices)
                plt.axvspan(min_idx-0.5, max_idx+0.5, color=colors[i % len(colors)], alpha=0.15, zorder=-100)
                plt.text((min_idx + max_idx)/2, plt.ylim()[1]*0.95, level.upper(), 
                        ha='center', fontweight='bold', alpha=0.7)
        
        # Add labels and legend
        plt.title("Effect of Neuron Ablation on Low-Level Speech Features", fontsize=14, fontweight='bold')
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Average Absolute Feature Change", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        
        # PLOT 2: Detailed heatmap visualization
        try:
            plt.subplot(2, 1, 2)
            
            # Create a heatmap matrix for visualization
            heatmap_data = np.zeros((len(layers), len(low_level_features)))
            
            # Fill the matrix with normalized values
            for i, layer_idx in enumerate(layers):
                for j, feat_name in enumerate(low_level_features):
                    if feature_changes[feat_name][i][0] > 0:  # Check if we have valid data
                        heatmap_data[i, j] = feature_changes[feat_name][i][0]
            
            # Normalize columns (features) for better visualization
            for j in range(heatmap_data.shape[1]):
                if np.max(heatmap_data[:, j]) > 0:
                    heatmap_data[:, j] = heatmap_data[:, j] / np.max(heatmap_data[:, j])
            
            # Plot heatmap
            ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd",
                           xticklabels=[feature_names_display[f] for f in low_level_features],
                           yticklabels=layers)
            
            plt.title("Normalized Impact of Layer Ablation on Speech Features", fontsize=14, fontweight='bold')
            plt.ylabel("Layer Index", fontsize=12)
            plt.xlabel("Feature", fontsize=12)
            
            # Add colorbar label
            cbar = ax.collections[0].colorbar
            cbar.set_label("Normalized Impact (0-1)", fontsize=10)
        except Exception as e:
            logger.error(f"Error creating heatmap visualization: {e}")
            plt.subplot(2, 1, 2)
            plt.text(0.5, 0.5, "Error creating heatmap visualization", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "low_level_feature_changes.png"), dpi=300)
        plt.close()
        
        logger.info(f"Low-level feature changes plot saved to {os.path.join(self.save_dir, 'low_level_feature_changes.png')}")

    def plot_temporal_spread(self, results):
        """
        Plot temporal spread of ablation effects
        """
        # Set up figure
        plt.figure(figsize=(12, 10))
        
        # Get layers in order
        layers = sorted(results['ablated'].keys())
        
        # Create a matrix to store L2 differences
        l2_diffs = []
        
        # For each layer being ablated
        for layer_idx in layers:
            layer_diffs = []
            
            try:
                # Get results for this layer
                layer_result = results['ablated'][layer_idx]
                
                # Check if temporal_spread exists and has l2_diff_per_layer
                if ('temporal_spread' not in layer_result or 
                    'l2_diff_per_layer' not in layer_result['temporal_spread']):
                    l2_diffs.append([0.0])
                    continue
                    
                l2_diff_per_layer = layer_result['temporal_spread']['l2_diff_per_layer']
                
                # For each potentially affected layer, store mean L2 diff
                for affected_layer in sorted(l2_diff_per_layer.keys()):
                    try:
                        diff = l2_diff_per_layer[affected_layer]
                        mean_diff = diff.mean().item() if hasattr(diff, 'mean') else float(diff)
                        layer_diffs.append(mean_diff)
                    except Exception as e:
                        logger.error(f"Error processing affected layer {affected_layer}: {e}")
                        layer_diffs.append(0.0)
                    
                l2_diffs.append(layer_diffs)
            except Exception as e:
                logger.error(f"Error processing layer {layer_idx} for temporal spread: {e}")
                l2_diffs.append([0.0])
        
        # Convert to numpy array and handle empty case
        if not l2_diffs or not any(l2_diffs):
            logger.warning("No temporal spread data to plot")
            plt.text(0.5, 0.5, "No temporal spread data available", 
                   ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
        else:
            try:
                # Create a matrix of consistent size
                max_layers = max(len(diff) for diff in l2_diffs)
                matrix = np.zeros((len(layers), max_layers))
                
                for i, diff in enumerate(l2_diffs):
                    matrix[i, :len(diff)] = diff
                
                # Plot heatmap
                im = plt.imshow(matrix, aspect='auto', cmap='viridis')
                plt.colorbar(im, label='Mean L2 Difference')
                plt.xlabel("Affected Layer")
                plt.ylabel("Ablated Layer")
                plt.title("Temporal Spread of Ablation Effects")
                
                # Set tick labels
                plt.yticks(range(len(layers)), layers)
                
                # Use the same layers for x-ticks if appropriate
                if max_layers <= len(layers):
                    plt.xticks(range(max_layers), range(max_layers))
            except Exception as e:
                logger.error(f"Error plotting temporal spread heatmap: {e}")
                plt.text(0.5, 0.5, f"Error plotting temporal spread: {e}", 
                       ha='center', va='center', transform=plt.gca().transAxes)
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "temporal_spread.png"))
        plt.close()
        
        logger.info(f"Temporal spread plot saved to {os.path.join(self.save_dir, 'temporal_spread.png')}")

    def plot_upstream_norms(self, results):
        """
        Plot upstream norm changes for each layer
        """
        # Set up figure
        plt.figure(figsize=(10, 6))
        
        # Get layers in order
        layers = sorted(results['ablated'].keys())
        
        # Collect upstream norms
        norms = []
        valid_layers = []
        
        for layer_idx in layers:
            if layer_idx == 0:  # Skip first layer as it has no upstream
                norms.append(0)
                valid_layers.append(layer_idx)
                continue
            
            try:
                layer_result = results['ablated'][layer_idx]
                
                # Check if upstream_norm exists
                if 'upstream_norm' not in layer_result:
                    continue
                    
                norm = layer_result['upstream_norm']
                norms.append(float(norm))
                valid_layers.append(layer_idx)
            except Exception as e:
                logger.error(f"Error processing upstream norm for layer {layer_idx}: {e}")
        
        # Check if we have any valid data
        if not norms or not valid_layers:
            logger.warning("No valid upstream norm data to plot")
            plt.text(0.5, 0.5, "No upstream norm data available", 
                   ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
        else:
            # Plot bar chart
            plt.bar(valid_layers, norms, alpha=0.7)
            plt.xlabel("Layer Index")
            plt.ylabel("Upstream Change Norm")
            plt.title("Magnitude of Required Upstream Changes")
            plt.grid(True, axis='y', alpha=0.3)
            
            # Mark layer groups
            for level, layer_indices in self.layer_groups.items():
                for idx in layer_indices:
                    if idx in valid_layers:
                        plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
            
            # Add text labels above bars
            for i, v in enumerate(norms):
                plt.text(
                    valid_layers[i], 
                    v + 0.01, 
                    f"{v:.3f}",
                    ha='center',
                    fontsize=8,
                    rotation=90
                )
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "upstream_norms.png"))
        plt.close()
        
        logger.info(f"Upstream norms plot saved to {os.path.join(self.save_dir, 'upstream_norms.png')}")

    def plot_emotion_prob_changes(self, results):
        """
        Plot changes in emotion probabilities
        """
        # Set up figure
        plt.figure(figsize=(12, 8))
        
        # Get layers in order
        layers = sorted(results['ablated'].keys())
        
        # Get target class
        target_class = results['configs'].get('target_class', 0)
        
        # Track KL divergence across layers
        kl_divs = []
        valid_kl_layers = []
        
        # Track probability change for target class
        target_prob_changes = []
        valid_prob_layers = []
        
        # Collect changes for each layer
        for layer_idx in layers:
            try:
                layer_result = results['ablated'][layer_idx]
                
                # Get KL divergence if available
                if 'kl_divergence' in layer_result:
                    try:
                        kl_div = layer_result['kl_divergence'].mean().item()
                        kl_divs.append(kl_div)
                        valid_kl_layers.append(layer_idx)
                    except Exception as e:
                        logger.error(f"Error processing KL divergence for layer {layer_idx}: {e}")
                
                # Get emotion probability deltas if available
                if ('feature_deltas' in layer_result and 
                    'emotion_probs_delta' in layer_result['feature_deltas']):
                    try:
                        changes = layer_result['feature_deltas']['emotion_probs_delta']
                        
                        if changes is not None and target_class < changes.shape[1]:
                            # Get change for target class
                            target_change = changes[:, target_class].mean().item()
                            target_prob_changes.append(target_change)
                            valid_prob_layers.append(layer_idx)
                        else:
                            logger.warning(f"Target class {target_class} out of bounds for layer {layer_idx}")
                    except Exception as e:
                        logger.error(f"Error processing emotion probability deltas for layer {layer_idx}: {e}")
            except Exception as e:
                logger.error(f"Error processing layer {layer_idx} for emotion probability changes: {e}")
        
        # Plot KL divergence if available
        if kl_divs and valid_kl_layers:
            try:
                plt.subplot(2, 1, 1)
                plt.plot(valid_kl_layers, kl_divs, 'o-', color='blue')
                plt.xlabel("Layer Index")
                plt.ylabel("KL Divergence")
                plt.title("KL Divergence Between Original and Ablated Distributions")
                plt.grid(True, alpha=0.3)
                
                # Mark layer groups
                for level, layer_indices in self.layer_groups.items():
                    for idx in layer_indices:
                        if idx in valid_kl_layers:
                            plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
            except Exception as e:
                logger.error(f"Error plotting KL divergence: {e}")
                plt.subplot(2, 1, 1)
                plt.title("No KL Divergence Data Available")
        else:
            plt.subplot(2, 1, 1)
            plt.title("No KL Divergence Data Available")
            
        # Plot target class probability change if available
        if target_prob_changes and valid_prob_layers:
            try:
                plt.subplot(2, 1, 2)
                plt.bar(valid_prob_layers, target_prob_changes, alpha=0.7, color='orange')
                plt.xlabel("Layer Index")
                plt.ylabel("Probability Change")
                
                # Get emotion class name if available
                if target_class < len(self.emotion_classes):
                    emotion_name = self.emotion_classes[target_class]
                    plt.title(f"Change in Probability for Target Class: {emotion_name}")
                else:
                    plt.title(f"Change in Probability for Target Class: {target_class}")
                    
                plt.grid(True, axis='y', alpha=0.3)
                
                # Mark layer groups
                for level, layer_indices in self.layer_groups.items():
                    for idx in layer_indices:
                        if idx in valid_prob_layers:
                            plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
                            
                # Add zero line
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            except Exception as e:
                logger.error(f"Error plotting emotion probability changes: {e}")
                plt.subplot(2, 1, 2)
                plt.title("No Emotion Probability Change Data Available")
        else:
            plt.subplot(2, 1, 2)
            plt.title("No Emotion Probability Change Data Available")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "emotion_probability_changes.png"))
        plt.close()
        
        logger.info(f"Emotion probability changes plot saved to {os.path.join(self.save_dir, 'emotion_probability_changes.png')}")