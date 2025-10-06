import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import GradientShap
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import librosa
from torch.utils.data import Dataset, DataLoader
import logging
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cosine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('NeuronAblator')

class NeuronAblator:
    """
    A class for analyzing neuron importance in transformer-based speech models through
    ablation by sign flipping and upstream inversion.
    
    This class implements methods for:
        1. Extracting activations from model layers
        2. Computing important neurons using GradientSHAP
        3. Ablating neurons by flipping signs and optimizing upstream activations
        4. Computing low/mid/high-level speech features
        5. Analyzing and visualizing the effect of ablation on features
    """
    
    def __init__(
        self, 
        model,
        classifier_head,
        device='cuda',
        layer_groups={
            'low': [0, 1, 2],  # Low-level (acoustic/frame-local) processing
            'mid': [3, 4, 5],  # Mid-level (phonetic/prosodic) processing
            'high': [6, 7, 8]  # High-level (semantic/utterance) processing
        },
        sample_rate=16000,
        frame_length_ms=20,
        hop_length_ms=10,
        save_dir='./ablation_results'
    ):
        """
        Initialize the NeuronAblator.
        
        Args:
            model: Pretrained transformer model (e.g., WavLM)
            classifier_head: Task-specific classifier head (e.g., emotion classifier)
            device: Device to use ('cuda' or 'cpu')
            layer_groups: Dictionary mapping level names to layer indices
            sample_rate: Audio sample rate in Hz
            frame_length_ms: Frame length in milliseconds for feature extraction
            hop_length_ms: Hop length in milliseconds for feature extraction
            save_dir: Directory to save results
        """
        self.model = model
        self.classifier_head = classifier_head
        self.device = device
        self.layer_groups = layer_groups
        self.sample_rate = sample_rate
        
        # Convert ms to samples
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.hop_length = int(sample_rate * hop_length_ms / 1000)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        
        # Put models on device
        self.model.to(device)
        self.classifier_head.to(device)
        self.model.eval()
        self.classifier_head.eval()
        
        # For storing activations during forward pass
        self.activation_store = {}
        self._register_hooks()
        
        # Target emotion classes (for RAVDESS, adjust as needed)
        self.emotion_classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    def _register_hooks(self):
        """Register forward hooks to capture activations from each layer"""
        self.hooks = []
        
        # TODO: Adjust based on your model's architecture
        # This implementation assumes model.encoder.layers is a list of transformer layers
        for i in range(len(self.model.encoder.layers)):
            layer = self.model.encoder.layers[i]
            
            def get_hook(layer_idx):
                def hook(module, input, output):
                    self.activation_store[layer_idx] = output.detach()
                return hook
            
            # Register hook on the output of each layer
            h = layer.register_forward_hook(get_hook(i))
            self.hooks.append(h)
            
        logger.info(f"Registered hooks on {len(self.hooks)} layers")

    def remove_hooks(self):
        """Remove all registered hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        logger.info("Removed all hooks")

    def extract_activations(self, waveform_batch):
        """
        Extract activations from all layers for the given waveform batch.
        
        Args:
            waveform_batch: Tensor of shape [B, 1, T] containing audio waveforms
            
        Returns:
            Dictionary mapping layer indices to activation tensors [B, T_frames, D]
        """
        self.activation_store = {}
        
        # Forward pass through the model
        with torch.no_grad():
            # TODO: Adjust based on your model's input requirements
            features = self.model(waveform_batch)
            
        # Return a copy of the activations
        return {k: v.clone() for k, v in self.activation_store.items()}

    def run_from_layer(self, layer_idx, activations_l):
        """
        Run the model starting from the given layer with the provided activations.
        
        Args:
            layer_idx: Index of the starting layer
            activations_l: Activations tensor for the starting layer [B, T, D]
            
        Returns:
            Final output and intermediate activations
        """
        # Store original activations for later restoration
        original_activations = {k: v.clone() for k, v in self.activation_store.items()}
        
        # TODO: Adjust this implementation based on your model architecture
        # This is a placeholder implementation
        
        # Temporary storage for custom forward pass
        temp_activations = {layer_idx: activations_l}
        
        # Start from the given layer and process through the rest
        num_layers = len(self.model.encoder.layers)
        
        x = activations_l
        for i in range(layer_idx, num_layers):
            layer = self.model.encoder.layers[i]
            x = layer(x)
            temp_activations[i] = x.clone()
        
        # Process through any final layers and classifier
        output = self.classifier_head(x)
        
        # Restore original activations
        self.activation_store = original_activations
        
        return output, temp_activations

    def compute_grad_shap_topk(
        self, 
        activations, 
        labels, 
        target_class, 
        topk=50, 
        nsamples=50
    ):
        """
        Compute GradientSHAP attributions and return the top-k neurons per layer.
        
        Args:
            activations: Dict mapping layer indices to activation tensors [B, T, D]
            labels: Ground truth labels [B]
            target_class: Target class to explain
            topk: Number of top neurons to select
            nsamples: Number of samples for GradientSHAP
            
        Returns:
            Dictionary mapping layer indices to arrays of top-k neuron indices
        """
        logger.info(f"Computing GradientSHAP for target class: {target_class}")
        
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
            
            # Define a wrapper function that runs from this layer forward
            def layer_forward_wrapper(activations_input):
                # Reshape if needed
                if activations_input.shape != layer_acts.shape:
                    activations_input = activations_input.reshape(layer_acts.shape)
                
                # Run from this layer forward and get classifier output
                logits, _ = self.run_from_layer(layer_idx, activations_input)
                return logits[:, target_class]  # Return target class logit
            
            # Create baseline of zeros
            baseline = torch.zeros_like(layer_acts).to(self.device)
            
            # Initialize GradientSHAP
            grad_shap = GradientShap(layer_forward_wrapper)
            
            # Compute attributions (may need to process in smaller batches if memory constrained)
            attributions = grad_shap.attribute(
                layer_acts,
                baselines=baseline,
                n_samples=nsamples,
                stdevs=0.1
            )  # [n_samples, T, D]
            
            # Average attributions across samples and time
            # Take absolute value to consider both positive and negative importance
            attr_avg = torch.abs(attributions).mean(dim=0).mean(dim=0)  # [D]
            
            # Get top-k neurons
            _, top_neurons = torch.topk(attr_avg, min(topk, hidden_dim))
            
            result[layer_idx] = top_neurons.cpu().numpy()
            
        return result

    def flip_and_invert(
        self, 
        activations,
        layer_idx, 
        frame_idx, 
        neuron_idx, 
        mode='flip', 
        invert_steps=100, 
        lr=1e-2, 
        reg=1e-2,
        window=0
    ):
        """
        Flip neuron activation signs and invert upstream activations to be consistent.
        
        Args:
            activations: Dictionary of activations
            layer_idx: Layer to ablate
            frame_idx: Frame index (or indices) to ablate
            neuron_idx: Neuron index (or indices) to ablate
            mode: 'flip' to multiply by -1, 'zero' to set to 0
            invert_steps: Number of optimization steps for upstream inversion
            lr: Learning rate for optimizer
            reg: Regularization strength for upstream changes
            window: Window size around target frame (0 means single frame)
            
        Returns:
            Dictionary with original activations, ablated activations, and upstream optimized activations
        """
        logger.info(f"Ablating layer {layer_idx}, frame {frame_idx}, neuron {neuron_idx}")
        
        # Get original activations for the target layer
        orig_layer_act = activations[layer_idx]  # [B, T, D]
        batch_size, seq_len, hidden_dim = orig_layer_act.shape
        
        # Create a mask for the target frames and neurons
        frame_mask = torch.zeros((batch_size, seq_len, 1), device=self.device)
        
        # Handle both single frame and multiple frame cases
        if isinstance(frame_idx, int):
            frame_indices = [max(0, frame_idx - window), min(seq_len, frame_idx + window + 1)]
            frame_mask[:, frame_indices[0]:frame_indices[1], :] = 1.0
        else:  # List of frames
            for idx in frame_idx:
                indices = [max(0, idx - window), min(seq_len, idx + window + 1)]
                frame_mask[:, indices[0]:indices[1], :] = 1.0
                
        # Create neuron mask
        neuron_mask = torch.zeros((1, 1, hidden_dim), device=self.device)
        if isinstance(neuron_idx, int):
            neuron_mask[0, 0, neuron_idx] = 1.0
        else:  # List of neurons
            neuron_mask[0, 0, neuron_idx] = 1.0
            
        # Combine masks
        combined_mask = frame_mask * neuron_mask  # [B, T, D]
        
        # Create ablated activations by copying original
        ablated_act = orig_layer_act.clone()
        
        # Apply ablation
        if mode == 'flip':
            # Flip sign of target neurons at target frames
            ablated_act = orig_layer_act * (1 - 2 * combined_mask)
        elif mode == 'zero':
            # Zero out target neurons at target frames
            ablated_act = orig_layer_act * (1 - combined_mask)
        else:
            raise ValueError(f"Unsupported ablation mode: {mode}")
            
        # If this is the first layer or we don't need inversion, return early
        if layer_idx == 0 or not invert_steps:
            return {
                'original': orig_layer_act,
                'ablated': ablated_act,
                'upstream_optimized': None
            }
            
        # Get upstream activations (previous layer)
        upstream_act = activations[layer_idx - 1]  # [B, T, D_prev]
        
        # Create upstream tensor for optimization (requires_grad=True)
        upstream_optim = upstream_act.clone().detach().requires_grad_(True)
        
        # Set up optimizer for upstream activations
        optimizer = torch.optim.Adam([upstream_optim], lr=lr)
        
        # Target is our ablated activation
        target_act = ablated_act
        
        # Perform optimization to find upstream activations
        pbar = tqdm(range(invert_steps), desc="Optimizing upstream")
        for i in pbar:
            optimizer.zero_grad()
            
            # Run one layer forward from optimized upstream
            # TODO: Adjust based on your model architecture
            output_act = self.model.encoder.layers[layer_idx](upstream_optim)
            
            # Compute loss: MSE to target + regularization to stay close to original
            mse_loss = F.mse_loss(output_act * combined_mask, target_act * combined_mask)
            reg_loss = reg * F.mse_loss(upstream_optim, upstream_act)
            total_loss = mse_loss + reg_loss
            
            # Backpropagate
            total_loss.backward()
            optimizer.step()
            
            # Update progress bar
            if i % 10 == 0:
                pbar.set_postfix({
                    'mse_loss': mse_loss.item(), 
                    'reg_loss': reg_loss.item(),
                    'total': total_loss.item()
                })
                
        # Return results
        return {
            'original': orig_layer_act,
            'ablated': ablated_act,
            'upstream_optimized': upstream_optim.detach()
        }

    def compute_features(self, waveform_batch, activations=None, logits=None):
        """
        Compute the required speech features:
        - Low: RMS energy, spectral centroid
        - Mid: Pitch (F0), Prosodic slope (F0 delta)
        - High: Emotion probabilities, Speaker embedding similarity
        
        Args:
            waveform_batch: Audio waveform tensor [B, 1, T]
            activations: Dictionary of model activations (optional)
            logits: Model output logits (optional)
            
        Returns:
            Dictionary of computed features
        """
        batch_size = waveform_batch.shape[0]
        features = {}
        
        # Extract activations and logits if not provided
        if activations is None:
            activations = self.extract_activations(waveform_batch)
            
        if logits is None:
            with torch.no_grad():
                # Get the final layer activation
                final_layer_idx = max(self.activation_store.keys())
                final_act = self.activation_store[final_layer_idx]
                
                # Pass through classifier head
                logits = self.classifier_head(final_act)
        
        # --- Low-level features ---
        
        # 1. RMS Energy per frame
        features['rms_energy'] = self._compute_rms_energy(waveform_batch)
        
        # 2. Spectral centroid
        features['spectral_centroid'] = self._compute_spectral_centroid(waveform_batch)
        
        # --- Mid-level features ---
        
        # 3. Pitch contour (F0)
        features['pitch_f0'] = self._estimate_f0(waveform_batch)
        
        # 4. Prosodic slope (F0 delta)
        features['f0_delta'] = self._compute_f0_delta(features['pitch_f0'])
        
        # --- High-level features ---
        
        # 5. Emotion probabilities
        probs = F.softmax(logits, dim=-1)
        features['emotion_probs'] = probs.detach()
        
        # 6. Speaker embedding similarity
        features['speaker_embedding'] = self._compute_speaker_embedding(activations)
        features['speaker_similarity'] = torch.ones(batch_size, device=self.device)  # Placeholder: will be filled in comparison
        
        return features

    def _compute_rms_energy(self, waveform):
        """
        Compute Root Mean Square energy per frame
        
        Args:
            waveform: Audio waveform tensor [B, 1, T]
            
        Returns:
            RMS energy per frame [B, n_frames]
        """
        batch_size = waveform.shape[0]
        
        # Convert [B, 1, T] -> [B, T] for frame processing
        waveform = waveform.squeeze(1)
        
        # Compute using torchaudio's RMS function if available, otherwise manual
        frames = torchaudio.functional.frame(
            waveform, 
            frame_length=self.frame_length, 
            hop_length=self.hop_length
        )  # [B, n_frames, frame_length]
        
        # Calculate RMS energy
        energy = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-6)  # [B, n_frames]
        
        return energy

    def _compute_spectral_centroid(self, waveform):
        """
        Compute spectral centroid per frame
        
        Args:
            waveform: Audio waveform tensor [B, 1, T]
            
        Returns:
            Spectral centroid per frame [B, n_frames]
        """
        batch_size = waveform.shape[0]
        results = []
        
        # Process each item in batch (could be vectorized further)
        for i in range(batch_size):
            wave = waveform[i, 0].cpu().numpy()
            
            # Compute using librosa
            cent = librosa.feature.spectral_centroid(
                y=wave, 
                sr=self.sample_rate,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )[0]  # [n_frames]
            
            results.append(torch.tensor(cent, device=self.device))
            
        # Stack results and return
        spectral_centroids = torch.stack(results)  # [B, n_frames]
        return spectral_centroids

    def _estimate_f0(self, waveform):
        """
        Estimate fundamental frequency (F0) per frame
        
        Args:
            waveform: Audio waveform tensor [B, 1, T]
            
        Returns:
            F0 per frame [B, n_frames]
        """
        batch_size = waveform.shape[0]
        results = []
        
        # Process each item in batch
        for i in range(batch_size):
            wave = waveform[i, 0].cpu().numpy()
            
            # Compute using librosa's PYIN algorithm for better accuracy
            f0, voiced_flag, voiced_probs = librosa.pyin(
                wave,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            
            # Fill NaN values (unvoiced) with 0
            f0 = np.nan_to_num(f0)
            
            results.append(torch.tensor(f0, device=self.device))
            
        # Stack results and return
        pitch = torch.stack(results)  # [B, n_frames]
        return pitch

    def _compute_f0_delta(self, f0):
        """
        Compute F0 delta (rate of change in pitch)
        
        Args:
            f0: Pitch tensor [B, n_frames]
            
        Returns:
            F0 delta per frame [B, n_frames]
        """
        # Compute gradient of F0 using torch.gradient
        f0_delta = torch.gradient(f0, dim=1)[0]  # [B, n_frames]
        
        return f0_delta

    def _compute_speaker_embedding(self, activations):
        """
        Compute speaker embedding by pooling high-level activations
        
        Args:
            activations: Dictionary of activations
            
        Returns:
            Speaker embeddings [B, D]
        """
        # Use highest layer for speaker embedding
        high_layers = self.layer_groups['high']
        highest_layer = max(high_layers)
        
        # Get activations from highest layer
        high_act = activations[highest_layer]  # [B, T, D]
        
        # Mean pooling across time dimension
        speaker_emb = high_act.mean(dim=1)  # [B, D]
        
        return speaker_emb

    def _compute_cosine_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between embeddings
        
        Args:
            emb1, emb2: Embedding tensors [B, D]
            
        Returns:
            Cosine similarity [B]
        """
        # Normalize embeddings
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(emb1_norm * emb2_norm, dim=1)  # [B]
        
        return similarity

    def run_layerwise_ablation(
        self,
        dataset_subset, 
        target_class=None,
        topk=50, 
        frame_policy='peak_frames',
        ablate_mode='flip',
        invert=True,
        invert_steps=100,
        lr=1e-2,
        reg=1e-2,
        window=0,
        batch_size=8,
        num_batches=5
    ):
        """
        Run layerwise ablation and measure effects on features.
        
        Args:
            dataset_subset: Dataset or list of waveforms and labels
            target_class: Class to target (if None, uses labels from dataset)
            topk: Number of top neurons to ablate
            frame_policy: How to select frames ('peak_frames', 'all', or indices)
            ablate_mode: 'flip' or 'zero'
            invert: Whether to optimize upstream activations
            invert_steps: Number of steps for upstream optimization
            lr: Learning rate for optimization
            reg: Regularization strength
            window: Window size around target frames
            batch_size: Batch size for processing
            num_batches: Number of batches to process
            
        Returns:
            Dictionary of results including feature changes and metrics
        """
        logger.info(f"Starting layerwise ablation with topk={topk}, ablate_mode={ablate_mode}")
        
        # Create dataloader
        dataloader = self._create_dataloader(dataset_subset, batch_size=batch_size)
        
        # Initialize results
        results = {
            'baseline': {},
            'ablated': {},
            'feature_deltas': {},
            'upstream_norms': {},
            'temporal_spread': {},
            'kl_divergence': {},
            'configs': {
                'topk': topk,
                'ablate_mode': ablate_mode,
                'invert': invert,
                'frame_policy': frame_policy,
                'target_class': target_class
            }
        }
        
        # Process a subset of batches
        batch_count = 0
        all_activations = []
        all_features_baseline = []
        all_waveforms = []
        all_labels = []
        
        for batch in dataloader:
            if batch_count >= num_batches:
                break
                
            # Unpack batch
            waveforms, labels = batch
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)
            all_waveforms.append(waveforms.cpu())
            all_labels.append(labels.cpu())
            
            # Extract baseline activations
            activations = self.extract_activations(waveforms)
            all_activations.append({k: v.cpu() for k, v in activations.items()})
            
            # Compute baseline features
            with torch.no_grad():
                final_layer_idx = max(activations.keys())
                final_act = activations[final_layer_idx]
                logits = self.classifier_head(final_act)
                
                features = self.compute_features(waveforms, activations, logits)
                all_features_baseline.append({k: v.cpu() for k, v in features.items()})
            
            batch_count += 1
        
        # Compute GradientSHAP for important neurons
        all_waveforms = torch.cat(all_waveforms, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Merge all activations
        merged_activations = {}
        for layer_idx in all_activations[0].keys():
            merged_activations[layer_idx] = torch.cat([act[layer_idx] for act in all_activations], dim=0).to(self.device)
        
        # Get target class if not provided
        if target_class is None and len(all_labels) > 0:
            # Use most common class
            target_class = torch.bincount(all_labels.to(torch.long)).argmax().item()
            
        # Compute important neurons
        important_neurons = self.compute_grad_shap_topk(
            merged_activations, 
            all_labels.to(self.device), 
            target_class,
            topk=topk
        )
        
        # Save baseline features
        results['baseline']['features'] = all_features_baseline
        results['baseline']['waveforms'] = all_waveforms
        results['baseline']['labels'] = all_labels
        results['baseline']['important_neurons'] = important_neurons
        
        # Perform ablation layer by layer
        layer_indices = sorted(important_neurons.keys())
        
        for layer_idx in layer_indices:
            logger.info(f"Ablating layer {layer_idx}")
            
            # Get neurons to ablate
            neurons_to_ablate = important_neurons[layer_idx]
            
            # Determine which frames to ablate based on policy
            frames_to_ablate = self._select_frames(merged_activations, layer_idx, frame_policy)
            
            # Store ablation results for this layer
            layer_results = []
            
            # Process each batch again with ablation
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                # Unpack batch
                waveforms, labels = batch
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)
                
                # Extract baseline activations
                activations = self.extract_activations(waveforms)
                
                # Apply ablation
                ablation_result = self.flip_and_invert(
                    activations,
                    layer_idx,
                    frames_to_ablate,
                    neurons_to_ablate,
                    mode=ablate_mode,
                    invert_steps=invert_steps if invert else 0,
                    lr=lr,
                    reg=reg,
                    window=window
                )
                
                # Get original and ablated activations
                orig_act = ablation_result['original']
                ablated_act = ablation_result['ablated']
                upstream_opt = ablation_result['upstream_optimized']
                
                # Compute features and metrics for original
                with torch.no_grad():
                    # Run from original layer activations
                    orig_logits, _ = self.run_from_layer(layer_idx, orig_act)
                    orig_features = self.compute_features(waveforms, activations, orig_logits)
                    
                    # Run from ablated layer activations
                    ablated_logits, ablated_acts = self.run_from_layer(layer_idx, ablated_act)
                    ablated_features = self.compute_features(waveforms, ablated_acts, ablated_logits)
                    
                    # If upstream optimization was done, run from there too
                    if upstream_opt is not None:
                        # Start one layer earlier with optimized activations
                        upstream_logits, upstream_acts = self.run_from_layer(layer_idx-1, upstream_opt)
                        
                        # Add modified upstream activations to the ablated_acts
                        ablated_acts[layer_idx-1] = upstream_opt
                        
                        # Recompute features
                        upstream_features = self.compute_features(waveforms, upstream_acts, upstream_logits)
                    else:
                        upstream_features = None
                    
                    # Compute feature deltas
                    feature_deltas = self._compute_feature_deltas(orig_features, ablated_features)
                    
                    # Compute other metrics
                    kl_div = self._compute_kl_divergence(
                        F.softmax(orig_logits, dim=-1),
                        F.softmax(ablated_logits, dim=-1)
                    )
                    
                    # Compute L2 norm difference between layers
                    temporal_spread = self._compute_temporal_spread(activations, ablated_acts)
                    
                    # Compute upstream change norm if upstream optimization was done
                    if upstream_opt is not None and layer_idx > 0:
                        upstream_norm = torch.norm(
                            upstream_opt - activations[layer_idx-1],
                            dim=-1
                        ).mean().item()
                    else:
                        upstream_norm = 0.0
                    
                    # Store results for this batch
                    batch_result = {
                        'feature_deltas': {k: v.cpu() for k, v in feature_deltas.items()},
                        'kl_divergence': kl_div.cpu(),
                        'temporal_spread': {k: v.cpu() for k, v in temporal_spread.items()},
                        'upstream_norm': upstream_norm
                    }
                    
                    layer_results.append(batch_result)
            
            # Aggregate results across batches for this layer
            results['ablated'][layer_idx] = self._aggregate_layer_results(layer_results)
            
        return results

    def _create_dataloader(self, dataset_subset, batch_size=8):
        """
        Create a dataloader from the dataset subset
        """
        # Check if dataset_subset is already a DataLoader
        if isinstance(dataset_subset, DataLoader):
            return dataset_subset
            
        # Check if dataset_subset is a Dataset
        if isinstance(dataset_subset, Dataset):
            return DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)
            
        # Otherwise, assume it's a list of (waveform, label) tuples
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx]
                
        return DataLoader(SimpleDataset(dataset_subset), batch_size=batch_size, shuffle=False)

    def _select_frames(self, activations, layer_idx, frame_policy):
        """
        Select frames to ablate based on policy
        
        Args:
            activations: Dictionary of activations
            layer_idx: Current layer index
            frame_policy: How to select frames ('peak_frames', 'all', or indices)
            
        Returns:
            List of frame indices to ablate
        """
        layer_act = activations[layer_idx]  # [B, T, D]
        seq_len = layer_act.shape[1]
        
        if frame_policy == 'all':
            # Ablate all frames
            return list(range(seq_len))
        elif frame_policy == 'peak_frames':
            # Find frames with highest average activation
            mean_act = torch.mean(torch.abs(layer_act), dim=-1)  # [B, T]
            mean_across_batch = torch.mean(mean_act, dim=0)  # [T]
            
            # Get top 3 peaks
            _, peak_indices = torch.topk(mean_across_batch, min(3, seq_len))
            
            return peak_indices.cpu().tolist()
        elif isinstance(frame_policy, list) or isinstance(frame_policy, np.ndarray):
            # Use provided indices
            return frame_policy
        else:
            # Default to middle frame
            return [seq_len // 2]

    def _compute_feature_deltas(self, orig_features, ablated_features):
        """
        Compute deltas between original and ablated features
        """
        deltas = {}
        
        # Compute deltas for each feature
        for feat_name in orig_features:
            if feat_name == 'emotion_probs' or feat_name == 'speaker_embedding':
                # For these features, compute relative changes differently
                continue
                
            # Compute absolute difference
            delta = ablated_features[feat_name] - orig_features[feat_name]
            deltas[feat_name] = delta
            
        # Special handling for emotion probs and speaker embedding
        # For emotion probs, compute KL divergence
        orig_probs = orig_features['emotion_probs']
        ablated_probs = ablated_features['emotion_probs']
        deltas['emotion_probs_delta'] = ablated_probs - orig_probs
        
        # For speaker embedding, compute cosine similarity
        orig_emb = orig_features['speaker_embedding']
        ablated_emb = ablated_features['speaker_embedding']
        
        # Compute speaker similarity (cosine)
        sim = self._compute_cosine_similarity(orig_emb, ablated_emb)
        deltas['speaker_similarity'] = sim
        
        return deltas

    def _compute_kl_divergence(self, p, q):
        """
        Compute KL divergence between probability distributions
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        p = p + eps
        q = q + eps
        
        # Normalize
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        
        # Compute KL divergence: sum(p * log(p/q))
        kl = torch.sum(p * torch.log(p / q), dim=-1)
        
        return kl

    def _compute_temporal_spread(self, orig_acts, ablated_acts):
        """
        Compute how ablation affects subsequent layers (temporal spread)
        
        Args:
            orig_acts: Original activations
            ablated_acts: Ablated activations
            
        Returns:
            Dictionary with L2 differences per layer and threshold counts
        """
        results = {
            'l2_diff_per_layer': {},
            'threshold_counts': {}
        }
        
        # Set threshold as 10% of average activation norm
        threshold = 0.1
        
        # Compute L2 diff for each layer
        for layer_idx in sorted(orig_acts.keys()):
            if layer_idx not in ablated_acts:
                continue
                
            orig = orig_acts[layer_idx]
            ablated = ablated_acts[layer_idx]
            
            # Compute L2 norm of difference per frame
            l2_diff = torch.norm(ablated - orig, dim=-1)  # [B, T]
            
            # Store difference
            results['l2_diff_per_layer'][layer_idx] = l2_diff
            
            # Count frames exceeding threshold
            threshold_val = threshold * torch.norm(orig, dim=-1).mean()
            count = (l2_diff > threshold_val).float().sum(dim=-1)  # [B]
            results['threshold_counts'][layer_idx] = count
            
        return results

    def _aggregate_layer_results(self, layer_results):
        """
        Aggregate results across batches for a layer
        """
        if not layer_results:
            return {}
            
        # Initialize aggregated results
        agg_results = {
            'feature_deltas': {},
            'kl_divergence': [],
            'temporal_spread': {'l2_diff_per_layer': {}, 'threshold_counts': {}},
            'upstream_norm': []
        }
        
        # Aggregate KL divergence and upstream norm
        for res in layer_results:
            agg_results['kl_divergence'].append(res['kl_divergence'])
            agg_results['upstream_norm'].append(res['upstream_norm'])
        
        # Convert to tensors
        agg_results['kl_divergence'] = torch.cat(agg_results['kl_divergence'], dim=0)
        agg_results['upstream_norm'] = np.mean(agg_results['upstream_norm'])
        
        # Aggregate feature deltas
        feature_names = layer_results[0]['feature_deltas'].keys()
        for feat_name in feature_names:
            # Concatenate across batches
            feat_deltas = torch.cat(
                [res['feature_deltas'][feat_name] for res in layer_results], 
                dim=0
            )
            agg_results['feature_deltas'][feat_name] = feat_deltas
            
        # Aggregate temporal spread
        # For simplicity, we'll just use the last batch's results
        agg_results['temporal_spread'] = layer_results[-1]['temporal_spread']
        
        return agg_results

    def summarize_and_plot(self, results):
        """
        Generate summary and visualization plots from ablation results
        
        Args:
            results: Results dictionary from run_layerwise_ablation
            
        Returns:
            None (saves figures to self.save_dir)
        """
        logger.info(f"Generating plots in {self.save_dir}")
        
        # Extract configuration
        config = results['configs']
        topk = config['topk']
        ablate_mode = config['ablate_mode']
        target_class = config['target_class']
        
        # Create a summary file
        summary_path = os.path.join(self.save_dir, f"ablation_summary_class{target_class}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Neuron Ablation Summary\n")
            f.write(f"=====================\n\n")
            f.write(f"Configuration:\n")
            f.write(f"- Target class: {target_class} ({self.emotion_classes[target_class] if target_class < len(self.emotion_classes) else 'Unknown'})\n")
            f.write(f"- Top-K neurons: {topk}\n")
            f.write(f"- Ablation mode: {ablate_mode}\n")
            f.write(f"- Frame policy: {config['frame_policy']}\n")
            f.write(f"- Upstream inversion: {config['invert']}\n\n")
            
            # Write layer group information
            f.write(f"Layer groups:\n")
            for level, layers in self.layer_groups.items():
                f.write(f"- {level}: {layers}\n")
            f.write("\n")
            
            # Write important neurons information
            f.write(f"Important neurons:\n")
            for layer_idx, neurons in results['baseline']['important_neurons'].items():
                f.write(f"- Layer {layer_idx}: {neurons[:10]}...\n")
            f.write("\n")
            
            # Write metrics for each layer
            f.write(f"Layer-wise metrics:\n")
            for layer_idx in sorted(results['ablated'].keys()):
                layer_result = results['ablated'][layer_idx]
                kl_div = layer_result['kl_divergence'].mean().item()
                upstream_norm = layer_result['upstream_norm']
                
                f.write(f"- Layer {layer_idx}:\n")
                f.write(f"  - KL divergence: {kl_div:.4f}\n")
                f.write(f"  - Upstream norm: {upstream_norm:.4f}\n")
                
                # Get level for this layer
                level = None
                for lev, layers in self.layer_groups.items():
                    if layer_idx in layers:
                        level = lev
                        break
                
                if level:
                    f.write(f"  - Level: {level}\n")
                f.write("\n")
        
        # Plot 1: Feature changes across layers
        self._plot_feature_changes(results)
        
        # Plot 2: Temporal spread heatmap
        self._plot_temporal_spread(results)
        
        # Plot 3: Upstream norm changes
        self._plot_upstream_norms(results)
        
        # Plot 4: Emotion probability changes
        self._plot_emotion_prob_changes(results)
        
        # Save full results as pickle
        pickle_path = os.path.join(self.save_dir, f"ablation_results_class{target_class}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
            
        logger.info(f"Summary saved to {summary_path}")
        logger.info(f"Full results saved to {pickle_path}")

    def _plot_feature_changes(self, results):
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
            'spectral_centroid': [],
            'pitch_f0': [],
            'f0_delta': [],
            'speaker_similarity': []
        }
        
        # Collect changes for each layer
        for layer_idx in layers:
            layer_result = results['ablated'][layer_idx]
            
            # Process each feature
            for feat_name in feature_changes.keys():
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
        
        # Create x-axis for layers
        x = np.array(layers)
        
        # Plot each feature
        for i, (feat_name, changes) in enumerate(feature_changes.items()):
            means = np.array([c[0] for c in changes])
            stds = np.array([c[1] for c in changes])
            
            plt.subplot(2, 3, i+1)
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
        plt.subplot(2, 3, 6)
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

    def _plot_temporal_spread(self, results):
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
            
            # Get results for this layer
            layer_result = results['ablated'][layer_idx]
            
            # Get temporal spread information
            temporal_spread = layer_result['temporal_spread']
            l2_diff_per_layer = temporal_spread['l2_diff_per_layer']
            
            # For each potentially affected layer, store mean L2 diff
            for affected_layer in sorted(l2_diff_per_layer.keys()):
                diff = l2_diff_per_layer[affected_layer]
                mean_diff = diff.mean().item()
                layer_diffs.append(mean_diff)
                
            l2_diffs.append(layer_diffs)
        
        # Convert to numpy array
        l2_diffs = np.array(l2_diffs)
        
        # Get affected layers
        affected_layers = sorted(l2_diff_per_layer.keys())
        
        # Plot heatmap
        plt.subplot(1, 1, 1)
        im = plt.imshow(l2_diffs, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Mean L2 Difference')
        plt.xlabel("Affected Layer")
        plt.ylabel("Ablated Layer")
        plt.title("Temporal Spread of Ablation Effects")
        
        # Set tick labels
        plt.xticks(range(len(affected_layers)), affected_layers)
        plt.yticks(range(len(layers)), layers)
        
        # Add grid
        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "temporal_spread.png"))
        plt.close()

    def _plot_upstream_norms(self, results):
        """
        Plot upstream norm changes for each layer
        """
        # Set up figure
        plt.figure(figsize=(10, 6))
        
        # Get layers in order
        layers = sorted(results['ablated'].keys())
        
        # Collect upstream norms
        norms = []
        for layer_idx in layers:
            if layer_idx == 0:  # Skip first layer as it has no upstream
                norms.append(0)
                continue
                
            layer_result = results['ablated'][layer_idx]
            norm = layer_result['upstream_norm']
            norms.append(norm)
        
        # Plot bar chart
        plt.bar(layers, norms, alpha=0.7)
        plt.xlabel("Layer Index")
        plt.ylabel("Upstream Change Norm")
        plt.title("Magnitude of Required Upstream Changes")
        plt.grid(True, axis='y', alpha=0.3)
        
        # Mark layer groups
        for level, layer_indices in self.layer_groups.items():
            for idx in layer_indices:
                if idx in layers:
                    plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
        
        # Add text labels above bars
        for i, v in enumerate(norms):
            plt.text(
                layers[i], 
                v + 0.01, 
                f"{v:.3f}",
                ha='center',
                fontsize=8,
                rotation=90
            )
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "upstream_norms.png"))
        plt.close()

    def _plot_emotion_prob_changes(self, results):
        """
        Plot changes in emotion probabilities
        """
        # Set up figure
        plt.figure(figsize=(12, 8))
        
        # Get layers in order
        layers = sorted(results['ablated'].keys())
        
        # Get target class
        target_class = results['configs']['target_class']
        
        # Track KL divergence across layers
        kl_divs = []
        
        # Track probability change for target class
        target_prob_changes = []
        
        # Collect changes for each layer
        for layer_idx in layers:
            layer_result = results['ablated'][layer_idx]
            
            # Get KL divergence
            kl_div = layer_result['kl_divergence'].mean().item()
            kl_divs.append(kl_div)
            
            # Get emotion probability deltas
            changes = layer_result['feature_deltas'].get('emotion_probs_delta', None)
            
            if changes is not None and target_class < changes.shape[1]:
                # Get change for target class
                target_change = changes[:, target_class].mean().item()
                target_prob_changes.append(target_change)
            else:
                target_prob_changes.append(0)
        
        # Plot KL divergence
        plt.subplot(2, 1, 1)
        plt.plot(layers, kl_divs, 'o-', color='blue')
        plt.xlabel("Layer Index")
        plt.ylabel("KL Divergence")
        plt.title("KL Divergence Between Original and Ablated Distributions")
        plt.grid(True, alpha=0.3)
        
        # Mark layer groups
        for level, layer_indices in self.layer_groups.items():
            for idx in layer_indices:
                if idx in layers:
                    plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
        
        # Plot target class probability change
        plt.subplot(2, 1, 2)
        plt.bar(layers, target_prob_changes, alpha=0.7, color='orange')
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
                if idx in layers:
                    plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
                    
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "emotion_probability_changes.png"))
        plt.close()
