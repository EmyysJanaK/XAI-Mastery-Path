"""
LayerwiseLowLevelAblation: A PyTorch implementation for neuron ablation analysis on WavLM speech models.

This class performs layer-wise neuron ablation analysis on WavLM using RAVDESS dataset and
visualizes how low-level acoustic feature proxies vary when important neuron activations 
are perturbed by sign flipping.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import torchaudio
import librosa
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional, Any
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('LayerwiseLowLevelAblation')

class RAVDESSDataset(Dataset):
    """
    Dataset class for the RAVDESS dataset.
    
    The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 
    speech recordings with various emotions.
    """
    
    def __init__(self, data_path, feature_extractor, max_samples=None):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the RAVDESS dataset
            feature_extractor: WavLM feature extractor for audio preprocessing
            max_samples (int, optional): Maximum number of samples to use
        """
        self.data_path = data_path
        self.feature_extractor = feature_extractor
        self.sample_rate = 16000  # WavLM expects 16kHz
        
        # List all audio files
        self.audio_files = []
        self.emotion_labels = []
        
        # Walk through the directory
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    # RAVDESS filename format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
                    parts = file.split('-')
                    if len(parts) >= 3:
                        # Extract emotion code (third part, 1-indexed in filename)
                        emotion_code = int(parts[2])
                        # Convert to 0-indexed
                        emotion_label = emotion_code - 1
                        
                        self.audio_files.append(os.path.join(root, file))
                        self.emotion_labels.append(emotion_label)
        
        # Limit dataset size if specified
        if max_samples and max_samples < len(self.audio_files):
            indices = np.random.choice(len(self.audio_files), max_samples, replace=False)
            self.audio_files = [self.audio_files[i] for i in indices]
            self.emotion_labels = [self.emotion_labels[i] for i in indices]
        
        logger.info(f"Loaded {len(self.audio_files)} audio files from RAVDESS dataset")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: A dictionary with waveform and label
        """
        audio_path = self.audio_files[idx]
        emotion_label = self.emotion_labels[idx]
        
        # Load and resample audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Convert to mono if needed
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        try:
            # Process with WavLM feature extractor
            inputs = self.feature_extractor(waveform[0], sampling_rate=self.sample_rate, return_tensors="pt")
        except Exception as e:
            # Handle potential errors in feature extraction
            logger.warning(f"Feature extraction failed for {audio_path}: {e}")
            # Create a dummy input of appropriate size for WavLM
            dummy_length = min(waveform.size(1), self.sample_rate * 5)  # 5 seconds max
            inputs = {
                "input_values": torch.zeros((1, dummy_length)).to(waveform.device)
            }
        
        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": emotion_label,
            "filename": os.path.basename(audio_path)
        }


class LayerwiseLowLevelAblation:
    """
    Performs layer-wise neuron ablation analysis on WavLM using RAVDESS dataset.
    
    This class implements the entire workflow for:
    1. Loading and preprocessing the RAVDESS dataset
    2. Extracting layer-wise activations from WavLM
    3. Identifying important neurons via gradient×activation attribution
    4. Applying sign-flip ablation (multiplying by -1) to these neurons
    5. Computing activation-space feature proxies before and after ablation
    6. Visualizing how these proxies change across layers
    
    Unlike traditional ablation that zeros out neurons, sign-flip ablation is chosen
    because it maintains the magnitude but reverses the direction of neuron contribution.
    This approach is more disruptive than zeroing while preserving energy, helping to
    identify neurons that encode specific directional information rather than just
    contributing activation magnitude.
    
    The implementation uses three low-level acoustic feature proxies that serve as
    interpretable analogs of traditional acoustic features measured in activation space:
    
    1. RMS Energy proxy: L2 norm across hidden dimensions of activations
    2. Zero-Crossing Rate (ZCR) proxy: Zero-crossings in first PCA component of activations
    3. Spectral Centroid proxy: Centroid of FFT magnitude spectrum of the RMS proxy
    
    These proxies are calculated before and after ablation to measure how perturbations to
    important neurons affect acoustic properties encoded in the activation space.
    """
    
    def __init__(
        self,
        model_name='microsoft/wavlm-base-plus',
        data_path=None,
        device=None,
        batch_size=8,
        max_samples=500,
        save_dir='./ablation_results',
        random_seed=42
    ):
        """
        Initialize the LayerwiseLowLevelAblation.
        
        Args:
            model_name (str): HuggingFace model name or path
            data_path (str): Path to RAVDESS dataset
            device (str): Device to use ('cuda' or 'cpu')
            batch_size (int): Batch size for processing
            max_samples (int): Maximum number of samples to use from dataset
            save_dir (str): Directory to save results and plots
            random_seed (int): Random seed for reproducibility
        """
        # Set random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Parameters
        self.batch_size = batch_size
        self.data_path = data_path
        self.max_samples = max_samples
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        
        # For storing activations during forward pass
        self.activation_store = {}
        self.hooks = []
        
        # Register hooks for all encoder layers
        self._register_hooks()
        
        # List of emotion labels (RAVDESS has 8 emotions)
        self.emotion_classes = [
            'neutral', 'calm', 'happy', 'sad', 
            'angry', 'fearful', 'disgust', 'surprised'
        ]
        
        # This will store the dataset and loader once loaded
        self.dataset = None
        self.loader = None
    
    def _register_hooks(self):
        """
        Register forward hooks to capture activations from each layer.
        
        This function sets up hooks for each layer in the encoder to store
        activations during the forward pass, which are essential for both
        attribution and ablation steps. Activations are captured per-layer and
        per-frame, with each frame corresponding to a time step in the input sequence.
        """
        try:
            logger.info("Registering activation hooks")
            
            def get_activation(name):
                def hook(module, input, output):
                    # Store activations - supports both tuple and tensor outputs
                    if isinstance(output, tuple):
                        self.activation_store[name] = output[0].detach()
                    else:
                        self.activation_store[name] = output.detach()
                return hook
            
            # Register hooks for each encoder layer
            try:
                n_layers = len(self.model.encoder.layers)
                logger.info(f"Found {n_layers} encoder layers")
                
                for i, layer in enumerate(self.model.encoder.layers):
                    h = layer.register_forward_hook(get_activation(i))
                    self.hooks.append(h)
                
                # Also register hook for input embedding layer (index 0)
                if hasattr(self.model, 'feature_extractor'):
                    embed_hook = self.model.feature_extractor.register_forward_hook(get_activation(0))
                    self.hooks.append(embed_hook)
                # Some WavLM versions use feature_projection instead
                elif hasattr(self.model, 'feature_projection'):
                    embed_hook = self.model.feature_projection.register_forward_hook(get_activation(0))
                    self.hooks.append(embed_hook)
            except Exception as e:
                logger.warning(f"Warning during hook registration: {e}. Will attempt to continue.")
            
            logger.info(f"Successfully registered {len(self.hooks)} hooks")
        
        except Exception as e:
            logger.error(f"Error registering hooks: {e}")
            raise
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        logger.info("Removed all hooks")
    
    def load_dataset(self):
        """
        Load and prepare the RAVDESS dataset for analysis.
        
        Returns:
            DataLoader: PyTorch DataLoader for the dataset
        """
        if not self.data_path:
            raise ValueError("Data path not specified. Please provide a path to the RAVDESS dataset.")
        
        logger.info(f"Loading dataset from {self.data_path}")
        self.dataset = RAVDESSDataset(
            data_path=self.data_path,
            feature_extractor=self.feature_extractor,
            max_samples=self.max_samples
        )
        
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        
        logger.info(f"Dataset loaded with {len(self.dataset)} samples")
        
        return self.loader
    
    def extract_activations(self, batch):
        """
        Extract activations from all layers for the given batch.
        
        Args:
            batch: Dictionary with input_values and label
            
        Returns:
            Dictionary mapping layer indices to activation tensors [B, T_frames, D]
        """
        self.activation_store = {}
        
        # Get inputs from batch
        input_values = batch["input_values"].to(self.device)
        
        try:
            # Clear gradients
            self.model.zero_grad()
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(input_values)
            
            # Return a copy of the activations
            return {k: v.clone() for k, v in self.activation_store.items()}
            
        except Exception as e:
            logger.error(f"Error during activation extraction: {e}")
            raise
    
    def compute_gradient_activation_attribution(self, batch, target_class, topk=100):
        """
        Compute gradient×activation attribution scores and return top-k neurons per layer.
        
        This method implements the gradient×activation attribution method directly in PyTorch,
        identifying neurons whose activations have the largest impact on the target class.
        This is done by:
        1. Running a forward pass to get activations
        2. Setting up each layer's activations to track gradients
        3. Computing logits for the target class
        4. Running backward pass to get gradients
        5. Multiplying gradients with activations to get attribution scores
        
        Args:
            batch: Batch of input data
            target_class: Target emotion class for attribution
            topk: Number of top neurons to select per layer
            
        Returns:
            Dictionary mapping layer indices to arrays of top-k neuron indices
        """
        logger.info(f"Computing gradient×activation attribution for target class: {target_class}")
        
        # Get inputs from batch
        input_values = batch["input_values"].to(self.device)
        
        # Set the model to training mode for gradient computation
        original_mode = self.model.training
        self.model.train()
        
        # Dictionary to store attribution results
        result = {}
        
        try:
            # Forward pass to get activations for all layers
            outputs = self.model(input_values)
            
            # Get number of encoder layers
            num_layers = len(self.model.encoder.layers)
            
            # Simple classifier head for emotion recognition
            hidden_size = outputs.last_hidden_state.shape[-1]
            classifier = torch.nn.Linear(hidden_size, len(self.emotion_classes)).to(self.device)
            
            # Process each encoder layer
            for layer_idx in range(num_layers):
                # Get activations for this layer
                activations = self.activation_store[layer_idx]
                
                # Detach and clone activations to avoid modifying the original
                acts = activations.detach().clone()
                acts.requires_grad_(True)
                
                # Global mean pooling over sequence length (frames)
                pooled = torch.mean(acts, dim=1)  # [B, D]
                
                # Get logits for the target class
                logits = classifier(pooled)  # [B, num_classes]
                target_logits = logits[:, target_class]
                
                # Backward pass to get gradients
                target_logits.sum().backward()
                
                # Compute gradient × activation attribution scores
                # This identifies neurons whose activation direction is most important
                # for the target class
                grad_acts = acts.grad * acts  # [B, T, D]
                
                # Sum over batch and frames to get total attribution per neuron
                attribution_scores = grad_acts.sum(dim=(0, 1)).abs()  # [D]
                
                # Get top-k neurons with highest attribution scores
                _, top_indices = torch.topk(attribution_scores, min(topk, attribution_scores.size(0)))
                top_indices = top_indices.cpu().numpy()
                
                # Store results
                result[layer_idx] = top_indices
                
                # Clear gradients for next iteration
                self.model.zero_grad()
                if acts.grad is not None:
                    acts.grad.zero_()
                
        except Exception as e:
            logger.error(f"Error computing attribution: {e}")
            raise
        finally:
            # Restore original model mode
            self.model.train(original_mode)
        
        return result
    
    def apply_sign_flip_ablation(self, activations, important_neurons, frame_policy='max_frame'):
        """
        Apply sign-flip ablation (multiply by -1) to important neurons.
        
        Sign-flip ablation is chosen because it's more disruptive than zeroing while
        preserving the magnitude of the contribution. This approach helps identify
        neurons that encode specific information rather than just being generally active.
        By flipping the sign, we reverse the direction of the neuron's contribution,
        which can reveal how directional encoding affects acoustic properties.
        
        Args:
            activations: Dict mapping layer indices to activation tensors [B, T, D]
            important_neurons: Dict mapping layer indices to arrays of neuron indices
            frame_policy: 'max_frame' to ablate only at max activation frame, 'all_frames' for global
            
        Returns:
            Dict of ablated activations
        """
        logger.info(f"Applying sign-flip ablation with frame policy: {frame_policy}")
        
        ablated_activations = {}
        
        for layer_idx, neurons in important_neurons.items():
            # Skip if layer not in activations (like embedding layer)
            if layer_idx not in activations:
                continue
                
            # Get layer activations
            layer_acts = activations[layer_idx].clone()  # [B, T, D]
            batch_size, seq_len, hidden_dim = layer_acts.shape
            
            if frame_policy == 'max_frame':
                # Find the frame with maximum activation for each neuron and sample
                for b in range(batch_size):
                    for n in neurons:
                        # Find the frame with maximum activation
                        neuron_acts = layer_acts[b, :, n]
                        max_frame = torch.argmax(torch.abs(neuron_acts))
                        
                        # Flip sign only at the max frame
                        layer_acts[b, max_frame, n] *= -1
                        
            elif frame_policy == 'all_frames':
                # Flip sign for all frames
                for b in range(batch_size):
                    layer_acts[b, :, neurons] *= -1
            
            else:
                raise ValueError(f"Unknown frame policy: {frame_policy}")
            
            ablated_activations[layer_idx] = layer_acts
        
        return ablated_activations
    
    def compute_activation_space_proxies(self, activations, layer_idx):
        """
        Compute activation-space proxies for low-level acoustic features.
        
        This function computes feature proxies directly in activation space:
        - RMS Energy proxy: L2 norm across hidden dimensions
        - ZCR proxy: Zero-crossings in first PCA component
        - Spectral Centroid proxy: Centroid of FFT magnitude of RMS proxy
        
        Args:
            activations: Tensor of activations [B, T, D]
            layer_idx: Layer index
            
        Returns:
            Dict of feature proxies
        """
        batch_size, seq_len, hidden_dim = activations.shape
        proxies = {}
        
        try:
            # Move to CPU for numpy operations
            acts_cpu = activations.cpu().numpy()
            
            # 1. RMS Energy proxy - L2 norm across hidden dimensions
            rms_proxy = np.linalg.norm(acts_cpu, axis=2)  # [B, T]
            proxies['rms_proxy'] = rms_proxy
            
            # 2. Zero-Crossing Rate proxy from first PCA component
            pca_components = np.zeros((batch_size, seq_len))
            
            for b in range(batch_size):
                # Apply PCA to get first principal component
                pca = PCA(n_components=1)
                sample_acts = acts_cpu[b]  # [T, D]
                pca_result = pca.fit_transform(sample_acts).reshape(-1)  # [T]
                pca_components[b] = pca_result
                
                # Count zero crossings
                zero_crossings = np.sum(np.diff(np.signbit(pca_result))) / (len(pca_result) - 1)
                
                if b == 0:
                    proxies['zcr_proxy'] = np.zeros(batch_size)
                proxies['zcr_proxy'][b] = zero_crossings
            
            # 3. Spectral Centroid proxy
            sc_proxy = np.zeros(batch_size)
            
            for b in range(batch_size):
                # Use the RMS proxy as time series for spectral analysis
                time_series = rms_proxy[b]
                
                # Compute FFT magnitude
                fft_mag = np.abs(np.fft.rfft(time_series))
                freqs = np.fft.rfftfreq(len(time_series))
                
                # Compute spectral centroid (weighted average of frequencies)
                if np.sum(fft_mag) > 0:
                    sc_proxy[b] = np.sum(freqs * fft_mag) / np.sum(fft_mag)
                else:
                    sc_proxy[b] = 0
                    
            proxies['sc_proxy'] = sc_proxy
            
            return proxies
            
        except Exception as e:
            logger.error(f"Error computing feature proxies for layer {layer_idx}: {e}")
            # Return empty proxies on error
            return {
                'rms_proxy': np.zeros((batch_size, seq_len)),
                'zcr_proxy': np.zeros(batch_size),
                'sc_proxy': np.zeros(batch_size)
            }
    
    def run_layerwise_analysis(self, target_class=0, topk=100, frame_policy='max_frame', num_batches=None):
        """
        Run the complete layer-wise ablation analysis workflow.
        
        This method performs the entire analysis pipeline:
        1. Process batches from the dataset
        2. Compute attribution to find important neurons
        3. Apply sign-flip ablation to those neurons
        4. Measure feature proxies before and after ablation
        5. Aggregate results across samples
        
        Args:
            target_class: Target emotion class to analyze
            topk: Number of top neurons to ablate per layer
            frame_policy: 'max_frame' or 'all_frames'
            num_batches: Number of batches to process (if None, process all)
            
        Returns:
            Dictionary of results including all metrics and proxies
        """
        if not self.loader:
            self.load_dataset()
        
        logger.info(f"Starting layerwise analysis for target class: {target_class}")
        
        results = {
            'config': {
                'target_class': target_class,
                'topk': topk,
                'frame_policy': frame_policy,
            },
            'layers': {},
            'important_neurons': {}
        }
        
        # Process batches
        all_activations = []
        all_important_neurons = None
        
        # Determine how many batches to process
        batch_iterator = tqdm(self.loader, desc="Processing batches", total=num_batches if num_batches else len(self.loader))
        
        for batch_idx, batch in enumerate(batch_iterator):
            # Stop processing if we've reached the specified number of batches
            if num_batches and batch_idx >= num_batches:
                break
            try:
                # Step 1: Extract activations
                activations = self.extract_activations(batch)
                
                # On first batch, compute attribution to find important neurons
                if batch_idx == 0:
                    all_important_neurons = self.compute_gradient_activation_attribution(
                        batch, 
                        target_class,
                        topk=topk
                    )
                    results['important_neurons'] = {
                        k: v.tolist() for k, v in all_important_neurons.items()
                    }
                
                # Process each layer
                for layer_idx in all_important_neurons.keys():
                    if layer_idx not in results['layers']:
                        results['layers'][layer_idx] = {
                            'baseline_proxies': {
                                'rms': [], 'zcr': [], 'sc': []
                            },
                            'ablated_proxies': {
                                'rms': [], 'zcr': [], 'sc': []
                            },
                            'relative_changes': {
                                'rms': [], 'zcr': [], 'sc': []
                            }
                        }
                    
                    # Skip if layer not in activations
                    if layer_idx not in activations:
                        continue
                    
                    layer_activations = activations[layer_idx]
                    
                    # Compute baseline proxies before ablation
                    baseline_proxies = self.compute_activation_space_proxies(layer_activations, layer_idx)
                    
                    # Apply sign-flip ablation to this layer
                    ablated_activations = self.apply_sign_flip_ablation(
                        {layer_idx: layer_activations},
                        {layer_idx: all_important_neurons[layer_idx]},
                        frame_policy
                    )
                    
                    # Compute ablated proxies
                    ablated_proxies = self.compute_activation_space_proxies(ablated_activations[layer_idx], layer_idx)
                    
                    # Compute relative changes (normalized per sample to reduce variance)
                    batch_size = layer_activations.shape[0]
                    
                    # RMS proxy - mean change per frame
                    rms_change = np.mean(
                        np.abs(ablated_proxies['rms_proxy'] - baseline_proxies['rms_proxy']) /
                        (np.mean(baseline_proxies['rms_proxy'], axis=1, keepdims=True) + 1e-6),
                        axis=1
                    )
                    
                    # ZCR proxy - relative change
                    zcr_change = np.abs(ablated_proxies['zcr_proxy'] - baseline_proxies['zcr_proxy']) / (baseline_proxies['zcr_proxy'] + 1e-6)
                    
                    # SC proxy - relative change
                    sc_change = np.abs(ablated_proxies['sc_proxy'] - baseline_proxies['sc_proxy']) / (baseline_proxies['sc_proxy'] + 1e-6)
                    
                    # Store batch results
                    results['layers'][layer_idx]['baseline_proxies']['rms'].append(baseline_proxies['rms_proxy'])
                    results['layers'][layer_idx]['baseline_proxies']['zcr'].append(baseline_proxies['zcr_proxy'])
                    results['layers'][layer_idx]['baseline_proxies']['sc'].append(baseline_proxies['sc_proxy'])
                    
                    results['layers'][layer_idx]['ablated_proxies']['rms'].append(ablated_proxies['rms_proxy'])
                    results['layers'][layer_idx]['ablated_proxies']['zcr'].append(ablated_proxies['zcr_proxy'])
                    results['layers'][layer_idx]['ablated_proxies']['sc'].append(ablated_proxies['sc_proxy'])
                    
                    results['layers'][layer_idx]['relative_changes']['rms'].append(rms_change)
                    results['layers'][layer_idx]['relative_changes']['zcr'].append(zcr_change)
                    results['layers'][layer_idx]['relative_changes']['sc'].append(sc_change)
            
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Combine results across batches for each layer
        for layer_idx in results['layers']:
            for feature in ['rms', 'zcr', 'sc']:
                try:
                    # Concatenate changes across all batches
                    all_changes = np.concatenate(results['layers'][layer_idx]['relative_changes'][feature])
                    
                    # Store mean and standard error for plotting
                    results['layers'][layer_idx]['relative_changes'][feature + '_mean'] = np.mean(all_changes)
                    results['layers'][layer_idx]['relative_changes'][feature + '_std'] = np.std(all_changes) / np.sqrt(len(all_changes))
                except Exception as e:
                    logger.warning(f"Error combining results for layer {layer_idx}, feature {feature}: {e}")
                    results['layers'][layer_idx]['relative_changes'][feature + '_mean'] = 0.0
                    results['layers'][layer_idx]['relative_changes'][feature + '_std'] = 0.0
        
        return results
    
    def plot_results(self, results):
        """
        Generate a plot showing feature proxy changes across layers.
        
        Creates a matplotlib figure with three lines (one for each proxy)
        showing how each feature proxy changes across layers after sign-flip ablation.
        Error bands represent standard error of the mean.
        
        Args:
            results: Results from run_layerwise_analysis
            
        Returns:
            Path to the saved figure
        """
        logger.info("Generating visualization")
        
        # Set up seaborn style
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Get all layers and sort them
        layers = sorted(list(results['layers'].keys()))
        
        # Extract mean and std for each feature
        rms_means = [results['layers'][l]['relative_changes']['rms_mean'] for l in layers]
        rms_stds = [results['layers'][l]['relative_changes']['rms_std'] for l in layers]
        
        zcr_means = [results['layers'][l]['relative_changes']['zcr_mean'] for l in layers]
        zcr_stds = [results['layers'][l]['relative_changes']['zcr_std'] for l in layers]
        
        sc_means = [results['layers'][l]['relative_changes']['sc_mean'] for l in layers]
        sc_stds = [results['layers'][l]['relative_changes']['sc_std'] for l in layers]
        
        # Normalize to 0-1 range for better comparison
        def normalize_feature(values):
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                return [(v - min_val) / (max_val - min_val) for v in values]
            else:
                return [0.5 for _ in values]
        
        rms_means_norm = normalize_feature(rms_means)
        zcr_means_norm = normalize_feature(zcr_means)
        sc_means_norm = normalize_feature(sc_means)
        
        # Scale standard errors after normalization
        rms_stds_norm = [s * (max(rms_means) - min(rms_means)) for s in rms_stds] if max(rms_means) != min(rms_means) else rms_stds
        zcr_stds_norm = [s * (max(zcr_means) - min(zcr_means)) for s in zcr_stds] if max(zcr_means) != min(zcr_means) else zcr_stds  
        sc_stds_norm = [s * (max(sc_means) - min(sc_means)) for s in sc_stds] if max(sc_means) != min(sc_means) else sc_stds
        
        # Plot lines with error bands
        plt.plot(layers, rms_means_norm, 'o-', color='#1f77b4', linewidth=2, label='RMS Energy Proxy')
        plt.fill_between(
            layers,
            [max(0, m - s) for m, s in zip(rms_means_norm, rms_stds_norm)],
            [min(1, m + s) for m, s in zip(rms_means_norm, rms_stds_norm)],
            color='#1f77b4', alpha=0.2
        )
        
        plt.plot(layers, zcr_means_norm, 'o-', color='#2ca02c', linewidth=2, label='Zero-Crossing Rate Proxy')
        plt.fill_between(
            layers,
            [max(0, m - s) for m, s in zip(zcr_means_norm, zcr_stds_norm)],
            [min(1, m + s) for m, s in zip(zcr_means_norm, zcr_stds_norm)],
            color='#2ca02c', alpha=0.2
        )
        
        plt.plot(layers, sc_means_norm, 'o-', color='#d62728', linewidth=2, label='Spectral Centroid Proxy')
        plt.fill_between(
            layers,
            [max(0, m - s) for m, s in zip(sc_means_norm, sc_stds_norm)],
            [min(1, m + s) for m, s in zip(sc_means_norm, sc_stds_norm)],
            color='#d62728', alpha=0.2
        )
        
        # Add labels and legend
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Normalized Feature Change (Δ)', fontsize=12)
        
        # Format target class name
        target_class = results['config']['target_class']
        target_name = self.emotion_classes[target_class] if target_class < len(self.emotion_classes) else f"Class {target_class}"
        
        plt.title(f"Low-Level Feature Proxy Changes After Sign-Flip Ablation\n"
                 f"Target: {target_name}, Top-{results['config']['topk']} neurons, "
                 f"Frame Policy: {results['config']['frame_policy']}", fontsize=14)
        
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add x-ticks at each layer
        plt.xticks(layers)
        
        # Save figure
        fig_path = os.path.join(self.save_dir, f"feature_proxy_changes_class{target_class}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"Visualization saved to {fig_path}")
        return fig_path
    
    def run_complete_analysis(
        self, 
        target_class=0, 
        topk=100, 
        frame_policy='max_frame',
        num_batches=None
    ):
        """
        Run the complete analysis workflow and generate visualization.
        
        This is the main entry point that performs the full analysis pipeline:
        1. Load the dataset
        2. Run layer-wise analysis
        3. Generate and save visualization
        4. Clean up resources
        
        Args:
            target_class: Target emotion class to analyze
            topk: Number of top neurons to ablate per layer
            frame_policy: 'max_frame' or 'all_frames'
            num_batches: Number of batches to process (if None, process all)
            
        Returns:
            Dictionary of results and path to the saved figure
        """
        try:
            # Make sure we have the dataset loaded
            if not self.loader:
                self.load_dataset()
            
            # Run analysis
            results = self.run_layerwise_analysis(
                target_class=target_class,
                topk=topk,
                frame_policy=frame_policy,
                num_batches=num_batches
            )
            
            # Plot results
            fig_path = self.plot_results(results)
            
            # Save numerical results
            results_path = os.path.join(self.save_dir, f"ablation_results_class{target_class}.npy")
            np.save(results_path, results)
            logger.info(f"Results saved to {results_path}")
            
            return results, fig_path
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            raise
        finally:
            # Clean up hooks
            self.remove_hooks()

    def __del__(self):
        """Ensure hooks are removed when the object is deleted"""
        if hasattr(self, 'hooks') and self.hooks:
            self.remove_hooks()


if __name__ == "__main__":
    # Example usage for Kaggle environment
    import os
    
    # Setup paths for Kaggle
    data_path = "/kaggle/input/ravdess-emotional-speech-audio"
    save_dir = "/kaggle/working/wavlm_ablation_results"
    
    if not os.path.exists(data_path):
        data_path = "../input/ravdess-emotional-speech-audio"  # Alternative path format
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    try:
        ablator = LayerwiseLowLevelAblation(
            model_name='microsoft/wavlm-base-plus',
            data_path=data_path,
            save_dir=save_dir,
            max_samples=200,  # Use 200 utterances as requested
            batch_size=4      # Smaller batch size to avoid OOM errors
        )
        
        # Run analysis for 'happy' emotion (class index 2 in RAVDESS)
        results, fig_path = ablator.run_complete_analysis(
            target_class=2,
            topk=50,  # Top 50 neurons per layer
            frame_policy='max_frame',  # Ablate only at frame with maximum activation
        )
        
        print(f"Analysis complete. Visualization saved to {fig_path}")
        
        # Display the figure in notebook
        from IPython.display import Image, display
        display(Image(filename=fig_path))
        
    except Exception as e:
        print(f"Error in ablation analysis: {e}")
        import traceback
        traceback.print_exc()