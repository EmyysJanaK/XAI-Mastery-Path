"""
Attribution methods for identifying important neurons in speech models.
"""
import torch
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger('AttributionMethods')

class AttributionMethods:
    """
    Implementation of various attribution methods for identifying important neurons.
    
    This class provides four main attribution methods:
    1. Integrated Gradients
    2. Direct Gradient
    3. Ablation-based Attribution
    4. Activation-based Attribution (gradient-free)
    """
    
    def __init__(self, ablator):
        """
        Initialize the AttributionMethods.
        
        Args:
            ablator: Parent NeuronAblator instance
        """
        self.ablator = ablator
        self.model = ablator.model
        self.classifier_head = ablator.classifier_head
        self.device = ablator.device
        
    def compute_integrated_gradients_topk(
        self, 
        activations, 
        labels, 
        target_class, 
        topk=50, 
        n_steps=50
    ):
        """
        Compute Integrated Gradients attributions and return the top-k neurons per layer.
        
        Args:
            activations: Dict mapping layer indices to activation tensors [B, T, D]
            labels: Ground truth labels [B]
            target_class: Target class to explain
            topk: Number of top neurons to select
            n_steps: Number of steps for integration
            
        Returns:
            Dictionary mapping layer indices to arrays of top-k neuron indices
        """
        logger.info(f"Computing Integrated Gradients for target class: {target_class}")
        
        result = {}
        target_samples = torch.where(labels == target_class)[0]
        
        if len(target_samples) == 0:
            logger.warning(f"No samples found for target class {target_class}")
            return {}
        
        # Make sure we're in train mode for gradient computation
        original_model_mode = self.model.training
        original_classifier_mode = self.classifier_head.training
        self.model.train()
        self.classifier_head.train()
            
        # Process each layer
        for layer_idx in tqdm(activations.keys(), desc="Processing layers"):
            try:
                # Get activations for target class samples
                layer_acts = activations[layer_idx][target_samples].clone().detach()  # [n_samples, T, D]
                batch_size, seq_len, hidden_dim = layer_acts.shape
                
                # Create baseline of zeros
                baseline = torch.zeros_like(layer_acts).to(self.device)
                
                # Initialize integrated gradients
                integrated_grads = torch.zeros_like(layer_acts).to(self.device)
                
                # Compute path integral
                for step in range(n_steps):
                    # Interpolate between baseline and input
                    alpha = (step + 1) / n_steps
                    interpolated = baseline + alpha * (layer_acts - baseline)
                    interpolated = interpolated.clone().detach().requires_grad_(True)
                    
                    # Ensure we're tracking gradients
                    torch.set_grad_enabled(True)
                    
                    # Forward from this layer
                    logits, _ = self.ablator.run_from_layer(layer_idx, interpolated)
                    
                    # Get target class logit
                    target_logit = logits[:, target_class].sum()
                    
                    # Compute gradient
                    if interpolated.grad is not None:
                        interpolated.grad.zero_()
                    target_logit.backward(retain_graph=False)
                    
                    # Add to integrated gradients
                    if interpolated.grad is not None:
                        integrated_grads += interpolated.grad / n_steps
                    
                    # Clear memory
                    del logits, target_logit
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Multiply by input - baseline
                attributions = integrated_grads * (layer_acts - baseline)
                
                # Average attributions across samples and time
                # Take absolute value to consider both positive and negative importance
                attr_avg = torch.abs(attributions).mean(dim=0).mean(dim=0)  # [D]
                
                # Get top-k neurons
                _, top_neurons = torch.topk(attr_avg, min(topk, hidden_dim))
                
                result[layer_idx] = top_neurons.cpu().numpy()
                
            except Exception as e:
                logger.error(f"Error computing integrated gradients for layer {layer_idx}: {e}")
                # Fallback: random selection of neurons
                hidden_dim = activations[layer_idx].shape[-1]
                result[layer_idx] = np.random.choice(hidden_dim, size=min(topk, hidden_dim), replace=False)
                
        # Restore original model modes
        self.model.train(original_model_mode)
        self.classifier_head.train(original_classifier_mode)
            
        return result

    def compute_ablation_attribution_topk(
        self,
        activations,
        labels,
        target_class,
        topk=50
    ):
        """
        Compute neuron importance through ablation studies and return top-k neurons per layer.
        
        Args:
            activations: Dict mapping layer indices to activation tensors [B, T, D]
            labels: Ground truth labels [B]
            target_class: Target class to explain
            topk: Number of top neurons to select
            
        Returns:
            Dictionary mapping layer indices to arrays of top-k neuron indices
        """
        logger.info(f"Computing Ablation Attribution for target class: {target_class}")
        
        result = {}
        target_samples = torch.where(labels == target_class)[0]
        
        if len(target_samples) == 0:
            logger.warning(f"No samples found for target class {target_class}")
            return {}
        
        # Process each layer
        for layer_idx in tqdm(activations.keys(), desc="Processing layers"):
            # Get activations for target class samples
            layer_acts = activations[layer_idx][target_samples]  # [n_samples, T, D]
            batch_size, seq_len, hidden_dim = layer_acts.shape
            
            # For performance reasons, test a subset of neurons if there are too many
            test_neurons = min(hidden_dim, 200)
            if hidden_dim > test_neurons:
                # Sample neurons randomly
                neuron_indices = torch.randperm(hidden_dim)[:test_neurons].tolist()
            else:
                neuron_indices = list(range(hidden_dim))
                
            # Get original output for baseline comparison
            with torch.no_grad():
                orig_logits, _ = self.ablator.run_from_layer(layer_idx, layer_acts)
                orig_probs = torch.nn.functional.softmax(orig_logits, dim=-1)
                orig_target_prob = orig_probs[:, target_class].mean()
            
            # Initialize importance scores
            importances = torch.zeros(hidden_dim, device=self.device)
            
            # Test each neuron by ablating it
            for neuron_idx in tqdm(neuron_indices, desc=f"Testing neurons in layer {layer_idx}"):
                # Create ablated version (zero out the neuron)
                ablated_acts = layer_acts.clone()
                ablated_acts[:, :, neuron_idx] = 0
                
                # Forward pass with ablated activation
                with torch.no_grad():
                    ablated_logits, _ = self.ablator.run_from_layer(layer_idx, ablated_acts)
                    ablated_probs = torch.nn.functional.softmax(ablated_logits, dim=-1)
                    ablated_target_prob = ablated_probs[:, target_class].mean()
                
                # Compute importance as change in target probability
                importances[neuron_idx] = torch.abs(orig_target_prob - ablated_target_prob)
            
            # Get top-k neurons
            _, top_neurons = torch.topk(importances, min(topk, test_neurons))
            
            result[layer_idx] = top_neurons.cpu().numpy()
            
        return result
        
    def compute_gradient_topk(
        self,
        activations,
        labels,
        target_class,
        topk=50
    ):
        """
        Compute direct gradient attributions and return top-k neurons per layer.
        
        Args:
            activations: Dict mapping layer indices to activation tensors [B, T, D]
            labels: Ground truth labels [B]
            target_class: Target class to explain
            topk: Number of top neurons to select
            
        Returns:
            Dictionary mapping layer indices to arrays of top-k neuron indices
        """
        logger.info(f"Computing Direct Gradients for target class: {target_class}")
        
        result = {}
        target_samples = torch.where(labels == target_class)[0]
        
        if len(target_samples) == 0:
            logger.warning(f"No samples found for target class {target_class}")
            return {}
        
        # Make sure we're in train mode for gradient computation
        original_model_mode = self.model.training
        original_classifier_mode = self.classifier_head.training
        self.model.train()
        self.classifier_head.train()
            
        # Process each layer
        for layer_idx in tqdm(activations.keys(), desc="Processing layers"):
            try:
                # Get activations for target class samples
                layer_acts = activations[layer_idx][target_samples].clone().detach()  # [n_samples, T, D]
                layer_acts.requires_grad_(True)  # Enable gradients for this tensor
                
                # Create a fresh computational graph
                torch.set_grad_enabled(True)
                
                # Forward from this layer
                logits, _ = self.ablator.run_from_layer(layer_idx, layer_acts)
                
                # Get target class logit
                target_logit = logits[:, target_class].sum()
                
                # Compute gradient
                target_logit.backward(retain_graph=False)  # Don't retain graph to free memory
                
                # Verify we have gradients
                if layer_acts.grad is None:
                    raise ValueError("No gradients were computed")
                
                # Get gradients (importance)
                importance = torch.abs(layer_acts.grad)  # [B, T, D]
                
                # Average across batch and time
                neuron_importance = importance.mean(dim=0).mean(dim=0)  # [D]
                
                # Get top-k neurons
                _, top_neurons = torch.topk(neuron_importance, min(topk, neuron_importance.shape[0]))
                
                result[layer_idx] = top_neurons.cpu().numpy()
            except Exception as e:
                logger.error(f"Error computing gradients for layer {layer_idx}: {e}")
                # Fallback: random selection of neurons
                hidden_dim = activations[layer_idx].shape[-1]
                result[layer_idx] = np.random.choice(hidden_dim, size=min(topk, hidden_dim), replace=False)
                
            # Clear any stored gradients to avoid memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Restore original model modes
        self.model.train(original_model_mode)
        self.classifier_head.train(original_classifier_mode)
        
        return result

    def compute_activation_based_topk(
        self,
        activations,
        labels,
        target_class,
        topk=50
    ):
        """
        Compute neuron importance based on activation patterns only (no gradients needed).
        This uses a simple heuristic: neurons with higher mean activation values for the target class
        compared to other classes are more important for that class.
        
        Args:
            activations: Dict mapping layer indices to activation tensors [B, T, D]
            labels: Ground truth labels [B]
            target_class: Target class to explain
            topk: Number of top neurons to select
            
        Returns:
            Dictionary mapping layer indices to arrays of top-k neuron indices
        """
        logger.info(f"Computing activation-based importance for target class: {target_class}")
        
        result = {}
        
        # Get samples for target class
        target_samples = torch.where(labels == target_class)[0]
        
        # Get samples for other classes
        other_samples = torch.where(labels != target_class)[0]
        
        if len(target_samples) == 0:
            logger.warning(f"No samples found for target class {target_class}")
            return {}
            
        # Process each layer
        for layer_idx in tqdm(activations.keys(), desc="Processing layers"):
            try:
                # Get activations for target class samples and other class samples
                target_acts = activations[layer_idx][target_samples]  # [target_samples, T, D]
                
                # Compute mean activation per neuron for target class
                target_mean = torch.abs(target_acts).mean(dim=0).mean(dim=0)  # [D]
                
                if len(other_samples) > 0:
                    other_acts = activations[layer_idx][other_samples]  # [other_samples, T, D]
                    other_mean = torch.abs(other_acts).mean(dim=0).mean(dim=0)  # [D]
                    
                    # Compute importance as difference between target and other classes
                    # Higher values mean more important for target class
                    importance = target_mean - other_mean
                else:
                    # If no other class samples, just use target activations
                    importance = target_mean
                
                # Get top-k neurons
                hidden_dim = importance.shape[0]
                _, top_neurons = torch.topk(importance, min(topk, hidden_dim))
                
                result[layer_idx] = top_neurons.cpu().numpy()
                
            except Exception as e:
                logger.error(f"Error computing activation-based importance for layer {layer_idx}: {e}")
                # Fallback: random selection of neurons
                hidden_dim = activations[layer_idx].shape[-1]
                result[layer_idx] = np.random.choice(hidden_dim, size=min(topk, hidden_dim), replace=False)
        
        return result