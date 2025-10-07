"""
Speech feature extraction functionality for neuron ablation analysis.
"""
import torch
import numpy as np
import librosa
import logging

logger = logging.getLogger('SpeechFeatureExtractor')

class SpeechFeatureExtractor:
    """
    Extract and analyze speech features for neuron ablation.
    
    This class extracts low-level, mid-level, and high-level speech features:
    - Low: RMS energy, amplitude envelope, zero-crossing rate, spectral centroid
    - Mid: Pitch (F0), prosodic slope (F0 delta)
    - High: Emotion probabilities, speaker embedding similarity
    """
    
    def __init__(self, ablator):
        """
        Initialize the SpeechFeatureExtractor.
        
        Args:
            ablator: Parent NeuronAblator instance
        """
        self.ablator = ablator
        self.device = ablator.device
        self.sample_rate = ablator.sample_rate
        self.frame_length = ablator.frame_length
        self.hop_length = ablator.hop_length
        self.emotion_classes = ablator.emotion_classes
        
    def compute_features(self, waveform_batch, activations=None, logits=None):
        """
        Compute the required speech features:
        - Low: RMS energy, amplitude envelope, zero-crossing rate, spectral centroid
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
        
        try:
            # Extract activations and logits if not provided
            if activations is None:
                activations = self.ablator.extract_activations(waveform_batch)
                
            if logits is None:
                with torch.no_grad():
                    # Check if activation store has any keys
                    if len(activations) > 0:
                        # Get the final layer activation
                        final_layer_idx = max(activations.keys())
                        final_act = activations[final_layer_idx]
                        
                        # Pass through classifier head
                        logits = self.ablator.classifier_head(final_act)
                    else:
                        # Create dummy logits
                        logger.warning("No activations available, creating dummy logits")
                        logits = torch.zeros(batch_size, len(self.emotion_classes), device=self.device)
            
            # --- Low-level features ---
            features['rms_energy'] = self._compute_rms_energy(waveform_batch)
            features['amplitude_envelope'] = self._compute_amplitude_envelope(waveform_batch)
            features['zero_crossing_rate'] = self._compute_zero_crossing_rate(waveform_batch)
            features['spectral_centroid'] = self._compute_spectral_centroid(waveform_batch)
            
            # --- Mid-level features ---
            features['pitch_f0'] = self._estimate_f0(waveform_batch)
            features['f0_delta'] = self._compute_f0_delta(features['pitch_f0'])
            
            # --- High-level features ---
            # Emotion probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            features['emotion_probs'] = probs.detach()
            
            # Speaker embedding and similarity
            features['speaker_embedding'] = self._compute_speaker_embedding(activations)
            features['speaker_similarity'] = torch.ones(batch_size, device=self.device)  # Placeholder
            
            return features
        
        except Exception as e:
            logger.error(f"Error in compute_features: {e}")
            logger.warning("Returning minimal feature set")
            
            # Return minimal feature set to avoid crashes
            return {
                'rms_energy': torch.ones(batch_size, 10, device=self.device),
                'amplitude_envelope': torch.ones(batch_size, 10, device=self.device),
                'zero_crossing_rate': torch.ones(batch_size, 10, device=self.device),
                'spectral_centroid': torch.ones(batch_size, 10, device=self.device),
                'pitch_f0': torch.ones(batch_size, 10, device=self.device),
                'f0_delta': torch.zeros(batch_size, 10, device=self.device),
                'emotion_probs': torch.ones(batch_size, len(self.emotion_classes), device=self.device) / len(self.emotion_classes),
                'speaker_embedding': torch.ones(batch_size, 768, device=self.device),
                'speaker_similarity': torch.ones(batch_size, device=self.device)
            }

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
        
        try:
            # Try to use torchaudio.functional.frame if available
            try:
                import torchaudio
                frames = torchaudio.functional.frame(
                    waveform, 
                    frame_length=self.frame_length, 
                    hop_length=self.hop_length
                )  # [B, n_frames, frame_length]
                
                # Calculate RMS energy
                energy = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-6)  # [B, n_frames]
            except (ImportError, AttributeError):
                # Manual framing if torchaudio.functional.frame is not available
                n_frames = (waveform.shape[1] - self.frame_length) // self.hop_length + 1
                frames = torch.zeros(batch_size, n_frames, self.frame_length, device=waveform.device)
                
                for i in range(n_frames):
                    start = i * self.hop_length
                    end = start + self.frame_length
                    if end <= waveform.shape[1]:
                        frames[:, i, :] = waveform[:, start:end]
                
                # Calculate RMS energy
                energy = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-6)  # [B, n_frames]
                
            return energy
        except Exception as e:
            logger.error(f"Error computing RMS energy: {e}")
            # Return dummy values
            return torch.ones(batch_size, 10, device=self.device)

    def _compute_amplitude_envelope(self, waveform):
        """
        Compute amplitude envelope per frame (maximum absolute amplitude)
        
        Args:
            waveform: Audio waveform tensor [B, 1, T]
            
        Returns:
            Amplitude envelope per frame [B, n_frames]
        """
        batch_size = waveform.shape[0]
        
        # Convert [B, 1, T] -> [B, T] for frame processing
        waveform = waveform.squeeze(1)
        
        try:
            # Try to use torchaudio.functional.frame if available
            try:
                import torchaudio
                frames = torchaudio.functional.frame(
                    waveform, 
                    frame_length=self.frame_length, 
                    hop_length=self.hop_length
                )  # [B, n_frames, frame_length]
                
                # Calculate amplitude envelope (max absolute amplitude)
                env = torch.max(torch.abs(frames), dim=-1)[0]  # [B, n_frames]
            except (ImportError, AttributeError):
                # Manual framing if torchaudio.functional.frame is not available
                n_frames = (waveform.shape[1] - self.frame_length) // self.hop_length + 1
                frames = torch.zeros(batch_size, n_frames, self.frame_length, device=waveform.device)
                
                for i in range(n_frames):
                    start = i * self.hop_length
                    end = start + self.frame_length
                    if end <= waveform.shape[1]:
                        frames[:, i, :] = waveform[:, start:end]
                
                # Calculate amplitude envelope
                env = torch.max(torch.abs(frames), dim=-1)[0]  # [B, n_frames]
                
            return env
        except Exception as e:
            logger.error(f"Error computing amplitude envelope: {e}")
            # Return dummy values
            return torch.ones(batch_size, 10, device=self.device)

    def _compute_zero_crossing_rate(self, waveform):
        """
        Compute zero crossing rate per frame
        
        Args:
            waveform: Audio waveform tensor [B, 1, T]
            
        Returns:
            Zero crossing rate per frame [B, n_frames]
        """
        batch_size = waveform.shape[0]
        results = []
        
        try:
            # Process each item in batch
            for i in range(batch_size):
                wave = waveform[i, 0].cpu().numpy()
                
                # Compute using librosa
                zcr = librosa.feature.zero_crossing_rate(
                    y=wave,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length
                )[0]  # [n_frames]
                
                results.append(torch.tensor(zcr, device=self.device))
                
            # Stack results and return
            zcr_tensor = torch.stack(results)  # [B, n_frames]
            return zcr_tensor
        except Exception as e:
            logger.error(f"Error computing zero crossing rate: {e}")
            # Return dummy values
            return torch.ones(batch_size, 10, device=self.device)

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
        
        try:
            # Process each item in batch
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
        except Exception as e:
            logger.error(f"Error computing spectral centroid: {e}")
            # Return dummy values
            return torch.ones(batch_size, 10, device=self.device)

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
        
        try:
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
        except Exception as e:
            logger.error(f"Error estimating F0: {e}")
            # Return dummy values
            return torch.ones(batch_size, 10, device=self.device)

    def _compute_f0_delta(self, f0):
        """
        Compute F0 delta (rate of change in pitch)
        
        Args:
            f0: Pitch tensor [B, n_frames]
            
        Returns:
            F0 delta per frame [B, n_frames]
        """
        try:
            # Compute gradient of F0 using torch.gradient
            f0_delta = torch.gradient(f0, dim=1)[0]  # [B, n_frames]
            return f0_delta
        except Exception as e:
            logger.error(f"Error computing F0 delta: {e}")
            # Return dummy values
            batch_size = f0.shape[0]
            return torch.zeros(batch_size, f0.shape[1], device=self.device)

    def _compute_speaker_embedding(self, activations):
        """
        Compute speaker embedding by pooling high-level activations
        
        Args:
            activations: Dictionary of activations
            
        Returns:
            Speaker embeddings [B, D]
        """
        try:
            # Use highest layer for speaker embedding
            high_layers = self.ablator.layer_groups['high']
            
            # Find the highest available layer
            available_high_layers = [layer for layer in high_layers if layer in activations]
            
            if available_high_layers:
                # Use the highest available layer
                highest_layer = max(available_high_layers)
                
                # Get activations from highest layer
                high_act = activations[highest_layer]  # [B, T, D]
                
                # Mean pooling across time dimension
                speaker_emb = high_act.mean(dim=1)  # [B, D]
                
                return speaker_emb
            else:
                # If no high layers are available, use the highest available layer
                if activations:
                    highest_available = max(activations.keys())
                    
                    high_act = activations[highest_available]  # [B, T, D]
                    speaker_emb = high_act.mean(dim=1)  # [B, D]
                    
                    return speaker_emb
                else:
                    # If no activations at all, return a dummy tensor
                    return torch.ones(1, 768, device=self.device)  # Dummy embedding
        except Exception as e:
            logger.error(f"Error in _compute_speaker_embedding: {e}")
            # Return a dummy embedding
            return torch.ones(1, 768, device=self.device)

    def _compute_cosine_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between embeddings
        
        Args:
            emb1, emb2: Embedding tensors [B, D]
            
        Returns:
            Cosine similarity [B]
        """
        # Normalize embeddings
        emb1_norm = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2_norm = torch.nn.functional.normalize(emb2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(emb1_norm * emb2_norm, dim=1)  # [B]
        
        return similarity
        
    def _compute_feature_deltas(self, orig_features, ablated_features):
        """
        Compute deltas between original and ablated features
        """
        deltas = {}
        
        # Compute deltas for each feature
        for feat_name in orig_features:
            if feat_name in ('emotion_probs', 'speaker_embedding'):
                continue
                
            # Compute absolute difference
            delta = ablated_features[feat_name] - orig_features[feat_name]
            deltas[feat_name] = delta
            
        # Special handling for emotion probs and speaker embedding
        # For emotion probs, compute delta
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
        
        try:
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
        except Exception as e:
            logger.error(f"Error in _compute_temporal_spread: {e}")
            # Return minimal results
            results = {
                'l2_diff_per_layer': {0: torch.tensor(0.0)},
                'threshold_counts': {0: torch.tensor(0.0)}
            }
            
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
        
        try:
            # Aggregate KL divergence and upstream norm
            for res in layer_results:
                if 'kl_divergence' in res:
                    agg_results['kl_divergence'].append(res['kl_divergence'])
                if 'upstream_norm' in res:
                    agg_results['upstream_norm'].append(res['upstream_norm'])
            
            # Convert to tensors if we have any data
            if agg_results['kl_divergence']:
                try:
                    agg_results['kl_divergence'] = torch.cat(agg_results['kl_divergence'], dim=0)
                except Exception as e:
                    logger.error(f"Error aggregating kl_divergence: {e}")
                    agg_results['kl_divergence'] = torch.tensor(0.0)
            else:
                agg_results['kl_divergence'] = torch.tensor(0.0)
                
            if agg_results['upstream_norm']:
                agg_results['upstream_norm'] = np.mean(agg_results['upstream_norm'])
            else:
                agg_results['upstream_norm'] = 0.0
            
            # Aggregate each feature across batches
            if all('feature_deltas' in res for res in layer_results) and layer_results:
                # Get all feature names across all batches
                all_feature_names = set()
                for res in layer_results:
                    all_feature_names.update(res['feature_deltas'].keys())
                
                # Aggregate each feature
                for feat_name in all_feature_names:
                    try:
                        # Collect all tensors for this feature
                        feat_tensors = [res['feature_deltas'][feat_name] for res in layer_results 
                                      if 'feature_deltas' in res and feat_name in res['feature_deltas']]
                        
                        # Only concatenate if we have any tensors
                        if feat_tensors:
                            feat_deltas = torch.cat(feat_tensors, dim=0)
                            agg_results['feature_deltas'][feat_name] = feat_deltas
                    except Exception as e:
                        logger.error(f"Error aggregating feature {feat_name}: {e}")
                        agg_results['feature_deltas'][feat_name] = torch.tensor(0.0)
            
            # Aggregate temporal spread - use the last batch that has temporal_spread
            for res in reversed(layer_results):
                if 'temporal_spread' in res and res['temporal_spread']:
                    agg_results['temporal_spread'] = res['temporal_spread']
                    break
        except Exception as e:
            logger.error(f"Error in _aggregate_layer_results: {e}")
            # Return minimal valid structure
            agg_results = {
                'feature_deltas': {},
                'kl_divergence': torch.tensor(0.0),
                'temporal_spread': {'l2_diff_per_layer': {}, 'threshold_counts': {}},
                'upstream_norm': 0.0
            }
        
        return agg_results