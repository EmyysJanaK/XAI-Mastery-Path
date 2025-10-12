   

"""
LayerwiseSignFlipAblation.py: Implementation of layer-wise sign-flipping ablation
for transformer-based speech models like WavLM.

This module provides a comprehensive class for performing sign-flipping ablation studies
across transformer layers to measure the impact on downstream speech tasks.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Callable
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor
from torch.utils.data import DataLoader


class LayerwiseSignFlipAblation:
    """
    A class for conducting layer-wise sign-flipping ablation studies on transformer-based 
    speech models like WavLM.
    
    The ablation is performed by flipping the sign of activations at specific layers and 
    measuring the resulting performance drop on downstream tasks.
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/wavlm-base-plus',
        device: Optional[str] = None,
        ser_probe_path: Optional[str] = None,
        sid_probe_path: Optional[str] = None,
        batch_size: int = 16,
        seed: int = 42
    ):
        """
        Initialize the layer-wise sign-flip ablation study.
        
        Args:
            model_name (str): HuggingFace model name for the speech model
            device (str, optional): Device to use ('cuda' or 'cpu')
            ser_probe_path (str, optional): Path to the trained SER probe
            sid_probe_path (str, optional): Path to the trained SID probe
            batch_size (int): Batch size for evaluation
            seed (int): Random seed for reproducibility
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = self._load_model(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get the number of layers in the model
        self.num_layers = len(self.model.encoder.layers)
        print(f"Model has {self.num_layers} layers")
        
        # Load probing heads
        self.ser_probe = self._load_probe(ser_probe_path, "SER")
        self.sid_probe = self._load_probe(sid_probe_path, "SID")
        
        # Keep track of the current hook handle for removal
        self.current_hook = None
        
        # Results storage
        self.baseline_results = {}
        self.ablation_results = {}
        
    def _load_model(self, model_name: str) -> torch.nn.Module:
        """
        Load the transformer model from HuggingFace.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            torch.nn.Module: Loaded model
        """
        try:
            model = AutoModel.from_pretrained(model_name)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_probe(self, probe_path: Optional[str], probe_type: str) -> torch.nn.Module:
        """
        Load a trained probing head.
        
        Args:
            probe_path (str, optional): Path to the trained probe
            probe_type (str): Type of probe ('SER' or 'SID')
            
        Returns:
            torch.nn.Module: Loaded probing head
        """
        if probe_path is None:
            print(f"No {probe_type} probe provided, creating a dummy probe for demonstration")
            # For SER: 8 emotions classification
            # For SID: Let's assume 20 speakers for demonstration
            num_classes = 8 if probe_type == "SER" else 20
            hidden_size = self.model.config.hidden_size
            probe = torch.nn.Linear(hidden_size, num_classes)
        else:
            try:
                probe = torch.load(probe_path, map_location=self.device)
                print(f"Loaded {probe_type} probe from {probe_path}")
            except Exception as e:
                print(f"Error loading {probe_type} probe: {e}")
                raise
        
        probe.to(self.device)
        probe.eval()
        return probe
    
    def sign_flip_hook(self, layer_idx: int) -> Callable:
        """
        Create a hook function that flips the sign of activations for a specific layer.
        
        The hook is applied after the GELU activation in the specified transformer layer.
        
        Args:
            layer_idx (int): Index of the layer to apply the sign-flip to
            
        Returns:
            Callable: Hook function that flips the sign of activations
        """
        def hook(module, input_tensor, output_tensor):
            """
            Hook function that flips the sign of the output tensor.
            
            Args:
                module: The PyTorch module (layer)
                input_tensor: Input to the module
                output_tensor: Output from the module
                
            Returns:
                Modified output tensor with flipped signs
            """
            # Check if output is a tuple (common in transformer layers)
            if isinstance(output_tensor, tuple):
                # Create a new tuple with the first element sign-flipped
                flipped_output = -output_tensor[0]
                return (flipped_output,) + output_tensor[1:]
            else:
                # Simple case: just flip the sign
                return -output_tensor
            
        print(f"Creating sign-flip hook for layer {layer_idx}")
        return hook
    
    def register_hook(self, layer_idx: int) -> None:
        """
        Register a sign-flipping hook for a specific layer.
        
        Args:
            layer_idx (int): Index of the layer to apply the hook to
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} is out of bounds [0, {self.num_layers-1}]")
        
        # Remove existing hook if present
        self.remove_hook()
        
        # Register hook after the GELU activation in the specified layer
        # In WavLM, this is typically in the feed-forward network part
        try:
            # Try to get the GELU activation in the feed-forward network
            module = self.model.encoder.layers[layer_idx].feed_forward.intermediate_act_fn
            
            # If that fails, try alternative locations common in transformer architectures
            if module is None:
                module = self.model.encoder.layers[layer_idx].feed_forward.act
            
            # Register the hook
            self.current_hook = module.register_forward_hook(self.sign_flip_hook(layer_idx))
            print(f"Successfully registered sign-flip hook for layer {layer_idx}")
        except AttributeError:
            # Fallback: register the hook at the layer output
            print(f"Could not find GELU activation, registering hook at layer output")
            module = self.model.encoder.layers[layer_idx]
            self.current_hook = module.register_forward_hook(self.sign_flip_hook(layer_idx))
    
    def remove_hook(self) -> None:
        """Remove the currently registered hook if it exists."""
        if self.current_hook is not None:
            self.current_hook.remove()
            self.current_hook = None
            print("Removed existing hook")
    
    def run_evaluation(
        self, 
        dataloader: DataLoader,
        probe_type: str = "SER"
    ) -> float:
        """
        Run evaluation on the dataset using the specified probe.
        
        Args:
            dataloader (DataLoader): DataLoader for the evaluation dataset
            probe_type (str): Type of probe to use ('SER' or 'SID')
            
        Returns:
            float: Accuracy of the model on the dataset
        """
        probe = self.ser_probe if probe_type == "SER" else self.sid_probe
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {probe_type}"):
                # Get inputs and labels
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass through the model
                outputs = self.model(input_values=input_values)
                
                # Get the last layer hidden states
                last_hidden_state = outputs.last_hidden_state
                
                # Average pooling over time dimension to get utterance-level representation
                pooled_output = torch.mean(last_hidden_state, dim=1)
                
                # Forward pass through the probe
                logits = probe(pooled_output)
                
                # Calculate accuracy
                _, predicted = torch.max(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"{probe_type} Accuracy: {accuracy:.4f}")
        return accuracy
    
    def run_baseline_evaluation(self, ser_dataloader: DataLoader, sid_dataloader: DataLoader) -> Dict[str, float]:
        """
        Run baseline evaluation without any ablation.
        
        Args:
            ser_dataloader (DataLoader): DataLoader for SER evaluation
            sid_dataloader (DataLoader): DataLoader for SID evaluation
            
        Returns:
            Dict[str, float]: Dictionary with baseline accuracies for SER and SID
        """
        print("Running baseline evaluation...")
        
        # Make sure no hooks are registered
        self.remove_hook()
        
        # Evaluate SER
        ser_accuracy = self.run_evaluation(ser_dataloader, "SER")
        
        # Evaluate SID
        sid_accuracy = self.run_evaluation(sid_dataloader, "SID")
        
        # Store results
        self.baseline_results = {
            "SER": ser_accuracy,
            "SID": sid_accuracy
        }
        
        return self.baseline_results
    
    def run_ablation_study(
        self, 
        ser_dataloader: DataLoader, 
        sid_dataloader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Run the layer-wise sign-flipping ablation study.
        
        Args:
            ser_dataloader (DataLoader): DataLoader for SER evaluation
            sid_dataloader (DataLoader): DataLoader for SID evaluation
            
        Returns:
            Dict[str, List[float]]: Dictionary with accuracy drops for each layer for SER and SID
        """
        # Run baseline evaluation if not already done
        if not self.baseline_results:
            self.run_baseline_evaluation(ser_dataloader, sid_dataloader)
        
        # Initialize results storage
        ser_drops = []
        sid_drops = []
        
        # Iterate through each layer
        for layer_idx in range(self.num_layers):
            print(f"\n--- Layer {layer_idx} Ablation ---")
            
            # Register hook for the current layer
            self.register_hook(layer_idx)
            
            # Evaluate SER with the current layer ablated
            ser_accuracy = self.run_evaluation(ser_dataloader, "SER")
            
            # Evaluate SID with the current layer ablated
            sid_accuracy = self.run_evaluation(sid_dataloader, "SID")
            
            # Calculate accuracy drops
            ser_drop = self.baseline_results["SER"] - ser_accuracy
            sid_drop = self.baseline_results["SID"] - sid_accuracy
            
            # Store results
            ser_drops.append(ser_drop)
            sid_drops.append(sid_drop)
            
            # Remove hook
            self.remove_hook()
            
            print(f"Layer {layer_idx} - SER Drop: {ser_drop:.4f}, SID Drop: {sid_drop:.4f}")
        
        # Store ablation results
        self.ablation_results = {
            "SER": ser_drops,
            "SID": sid_drops
        }
        
        return self.ablation_results
    
    # <<< MODIFIED METHOD STARTS HERE >>>
    def visualize_results(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the ablation study results using separate subplots for each task.
        
        Args:
            save_path (str, optional): Path to save the visualization
        """
        if not self.ablation_results:
            raise ValueError("No ablation results to visualize. Run the ablation study first.")
        
        # Prepare data
        layer_indices = list(range(self.num_layers))
        ser_drops = self.ablation_results["SER"]
        sid_drops = self.ablation_results["SID"]
        
        # Create figure with two subplots, sharing the x-axis
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True)
        
        # --- Subplot 1: SER Accuracy Drop ---
        axes[0].plot(layer_indices, ser_drops, 'b-', marker='o', linewidth=2, label='SER Accuracy Drop')
        axes[0].axvspan(4, 10, alpha=0.2, color='blue', label='Hypothesized SER region (layers 5-10)')
        axes[0].set_ylabel('Accuracy Drop (Baseline - Ablated)', fontsize=14)
        axes[0].set_title('SER Ablation Results', fontsize=16)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].legend(fontsize=12)
        
        # Annotate max point for SER
        max_ser_idx = np.argmax(ser_drops)
        axes[0].annotate(f'Max SER Drop: {ser_drops[max_ser_idx]:.4f}',
                         xy=(max_ser_idx, ser_drops[max_ser_idx]),
                         xytext=(max_ser_idx, ser_drops[max_ser_idx] + 0.05),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         fontsize=12)
                         
        # --- Subplot 2: SID Accuracy Drop ---
        axes[1].plot(layer_indices, sid_drops, 'r-', marker='s', linewidth=2, label='SID Accuracy Drop')
        axes[1].axvspan(10, 12, alpha=0.2, color='red', label='Hypothesized SID region (layers 11-12)')
        axes[1].set_xlabel('Layer Index', fontsize=14)
        axes[1].set_ylabel('Accuracy Drop (Baseline - Ablated)', fontsize=14)
        axes[1].set_title('SID Ablation Results', fontsize=16)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend(fontsize=12)
        
        # Annotate max point for SID
        max_sid_idx = np.argmax(sid_drops)
        axes[1].annotate(f'Max SID Drop: {sid_drops[max_sid_idx]:.4f}',
                         xy=(max_sid_idx, sid_drops[max_sid_idx]),
                         xytext=(max_sid_idx, sid_drops[max_sid_idx] + 0.05),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         fontsize=12)
        
        # Set overall figure properties
        fig.suptitle('Layer-wise Sign-Flipping Ablation Study', fontsize=18, y=0.98)
        plt.xticks(layer_indices)
        
        # Add text summarizing the findings
        if max_ser_idx >= 4 and max_ser_idx <= 9:
            ser_conclusion = "✓ SER hypothesis confirmed"
        else:
            ser_conclusion = "✗ SER hypothesis not confirmed"
            
        if max_sid_idx >= 10:
            sid_conclusion = "✓ SID hypothesis confirmed"
        else:
            sid_conclusion = "✗ SID hypothesis not confirmed"
            
        plt.figtext(0.5, 0.01, f"{ser_conclusion} (max at layer {max_ser_idx+1})\n"
                                f"{sid_conclusion} (max at layer {max_sid_idx+1})",
                    ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Adjust layout to prevent overlap
        fig.tight_layout(rect=[0, 0.05, 1, 0.96]) # Adjust rect to make space for suptitle and figtext

        # Save figure if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        # Show plot
        plt.show()
    # <<< MODIFIED METHOD ENDS HERE >>>

    def get_results_summary(self) -> Dict:
        """
        Get a summary of the ablation study results.
        
        Returns:
            Dict: Summary of the results
        """
        if not self.ablation_results:
            raise ValueError("No ablation results to summarize. Run the ablation study first.")
        
        ser_drops = self.ablation_results["SER"]
        sid_drops = self.ablation_results["SID"]
        
        max_ser_idx = np.argmax(ser_drops)
        max_sid_idx = np.argmax(sid_drops)
        
        # Check if the hypotheses are confirmed
        ser_hypothesis_confirmed = 4 <= max_ser_idx <= 9  # layers 5-10 (0-indexed)
        sid_hypothesis_confirmed = max_sid_idx >= 10  # layers 11-12 (0-indexed)
        
        # Create summary
        summary = {
            "baseline": self.baseline_results,
            "accuracy_drops": self.ablation_results,
            "max_ser_drop": {
                "layer": max_ser_idx,
                "drop": ser_drops[max_ser_idx]
            },
            "max_sid_drop": {
                "layer": max_sid_idx,
                "drop": sid_drops[max_sid_idx]
            },
            "hypotheses": {
                "ser_confirmed": ser_hypothesis_confirmed,
                "sid_confirmed": sid_hypothesis_confirmed
            }
        }
        
        return summary
        
    def create_and_train_probes(self, 
                                  ser_train_dataloader: DataLoader,
                                  sid_train_dataloader: DataLoader,
                                  num_epochs: int = 5,
                                  learning_rate: float = 1e-4,
                                  ser_save_path: Optional[str] = None,
                                  sid_save_path: Optional[str] = None) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Create and train SER and SID probes using the pre-trained WavLM model.
        
        Args:
            ser_train_dataloader: DataLoader for SER training data
            sid_train_dataloader: DataLoader for SID training data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            ser_save_path: Path to save trained SER probe
            sid_save_path: Path to save trained SID probe
            
        Returns:
            Tuple containing trained SER and SID probes
        """
        # Get model's hidden size
        hidden_size = self.model.config.hidden_size
        
        # Create probes
        ser_probe = torch.nn.Linear(hidden_size, 8)  # 8 emotion classes
        sid_probe = torch.nn.Linear(hidden_size, 24)  # Assuming 24 speakers in RAVDESS
        
        ser_probe.to(self.device)
        sid_probe.to(self.device)
        
        # Set model to evaluation mode (freeze all parameters)
        self.model.eval()
        
        # Define loss function and optimizers
        criterion = torch.nn.CrossEntropyLoss()
        ser_optimizer = torch.optim.Adam(ser_probe.parameters(), lr=learning_rate)
        sid_optimizer = torch.optim.Adam(sid_probe.parameters(), lr=learning_rate)
        
        # Train SER probe
        print(f"Training SER probe for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            ser_probe.train()
            for batch_idx, batch in enumerate(tqdm(ser_train_dataloader, desc=f"SER Epoch {epoch+1}/{num_epochs}")):
                # Get inputs and labels
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Zero the parameter gradients
                ser_optimizer.zero_grad()
                
                # Forward pass through WavLM (no gradient calculation needed)
                with torch.no_grad():
                    outputs = self.model(input_values=input_values)
                    # Get the last hidden states and pool over time dimension
                    last_hidden = outputs.last_hidden_state
                    pooled_output = torch.mean(last_hidden, dim=1)
                
                # Forward pass through probe
                logits = ser_probe(pooled_output)
                
                # Calculate loss
                loss = criterion(logits, labels)
                
                # Backward pass and optimize
                loss.backward()
                ser_optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Print statistics every 10 batches
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(ser_train_dataloader):
                    print(f"[{epoch+1}, {batch_idx+1:5d}] loss: {running_loss/10:.3f}, acc: {100*correct/total:.2f}%")
                    running_loss = 0.0
        
        # Train SID probe
        print(f"Training SID probe for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            sid_probe.train()
            for batch_idx, batch in enumerate(tqdm(sid_train_dataloader, desc=f"SID Epoch {epoch+1}/{num_epochs}")):
                # Get inputs and labels
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Zero the parameter gradients
                sid_optimizer.zero_grad()
                
                # Forward pass through WavLM (no gradient calculation needed)
                with torch.no_grad():
                    outputs = self.model(input_values=input_values)
                    # Get the last hidden states and pool over time dimension
                    last_hidden = outputs.last_hidden_state
                    pooled_output = torch.mean(last_hidden, dim=1)
                
                # Forward pass through probe
                logits = sid_probe(pooled_output)
                
                # Calculate loss
                loss = criterion(logits, labels)
                
                # Backward pass and optimize
                loss.backward()
                sid_optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Print statistics every 10 batches
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(sid_train_dataloader):
                    print(f"[{epoch+1}, {batch_idx+1:5d}] loss: {running_loss/10:.3f}, acc: {100*correct/total:.2f}%")
                    running_loss = 0.0
        
        # Save trained probes if paths are provided
        if ser_save_path:
            torch.save(ser_probe, ser_save_path)
            print(f"SER probe saved to {ser_save_path}")
        
        if sid_save_path:
            torch.save(sid_probe, sid_save_path)
            print(f"SID probe saved to {sid_save_path}")
        
        # Set the trained probes as class attributes
        self.ser_probe = ser_probe
        self.sid_probe = sid_probe
        
        return ser_probe, sid_probe
    
    def create_ravdess_dataloaders(self,
                                   ravdess_path: str,
                                   batch_size: int = 16,
                                   train_ratio: float = 0.8,
                                   random_state: int = 42) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for the RAVDESS dataset for both SER and SID tasks.
        
        Args:
            ravdess_path: Path to the RAVDESS dataset
            batch_size: Batch size for DataLoaders
            train_ratio: Ratio of data to use for training
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train and test DataLoaders for SER and SID tasks
        """
        import os
        import librosa
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from torch.utils.data import Dataset, DataLoader
        import torch
        import re
        
        class RavdessDataset(Dataset):
            def __init__(self, audio_files, labels, feature_extractor=None, target_sr=16000, max_length=160000):
                self.audio_files = audio_files
                self.labels = labels
                self.feature_extractor = feature_extractor
                self.target_sr = target_sr
                self.max_length = max_length  # Set a fixed maximum length (e.g., 10 seconds at 16kHz)
                
            def __len__(self):
                return len(self.audio_files)
            
            def __getitem__(self, idx):
                audio_file = self.audio_files[idx]
                label = self.labels[idx]
                
                # Load audio
                waveform, sr = librosa.load(audio_file, sr=self.target_sr, mono=True)
                
                # Handle variable length by padding or truncating
                if len(waveform) > self.max_length:
                    # Truncate if too long
                    waveform = waveform[:self.max_length]
                elif len(waveform) < self.max_length:
                    # Pad with zeros if too short
                    padding = np.zeros(self.max_length - len(waveform))
                    waveform = np.concatenate((waveform, padding))
                
                # Process with feature extractor if provided
                if self.feature_extractor:
                    inputs = self.feature_extractor(waveform, sampling_rate=self.target_sr, return_tensors="pt")
                    return {
                        "input_values": inputs.input_values.squeeze(),
                        "labels": torch.tensor(label, dtype=torch.long)
                    }
                else:
                    # If no feature extractor, just return the waveform
                    return {
                        "input_values": torch.tensor(waveform, dtype=torch.float),
                        "labels": torch.tensor(label, dtype=torch.long)
                    }
        
        print(f"Loading RAVDESS dataset from {ravdess_path}")
        
        # Load feature extractor
        from transformers import Wav2Vec2FeatureExtractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        
        # Get all audio files
        audio_files = []
        for root, _, files in os.walk(ravdess_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            raise ValueError(f"No audio files found in {ravdess_path}")
        
        print(f"Found {len(audio_files)} audio files")
        
        # Parse metadata from filenames
        # RAVDESS filename format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
        # Example: 03-01-01-01-01-01-01.wav
        metadata = []
        for file in audio_files:
            filename = os.path.basename(file)
            parts = re.match(r'(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)\.wav', filename)
            
            if parts:
                modality = int(parts.group(1))
                vocal_channel = int(parts.group(2))
                emotion = int(parts.group(3))
                intensity = int(parts.group(4))
                statement = int(parts.group(5))
                repetition = int(parts.group(6))
                actor = int(parts.group(7))
                
                metadata.append({
                    'file': file,
                    'modality': modality,
                    'vocal_channel': vocal_channel,
                    'emotion': emotion - 1,  # Make 0-indexed
                    'intensity': intensity,
                    'statement': statement,
                    'repetition': repetition,
                    'actor': actor - 1  # Make 0-indexed
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(metadata)
        
        # Split into training and testing sets
        train_files, test_files = train_test_split(
            df, train_size=train_ratio, random_state=random_state, stratify=df[['emotion', 'actor']]
        )
        
        # Create datasets for SER (emotion recognition)
        ser_train_dataset = RavdessDataset(
            train_files['file'].values,
            train_files['emotion'].values,
            feature_extractor=feature_extractor
        )
        
        ser_test_dataset = RavdessDataset(
            test_files['file'].values,
            test_files['emotion'].values,
            feature_extractor=feature_extractor
        )
        
        # Create datasets for SID (speaker identification)
        sid_train_dataset = RavdessDataset(
            train_files['file'].values,
            train_files['actor'].values,
            feature_extractor=feature_extractor
        )
        
        sid_test_dataset = RavdessDataset(
            test_files['file'].values,
            test_files['actor'].values,
            feature_extractor=feature_extractor
        )
        
        # Create DataLoaders with appropriate collate function
        # Using num_workers=0 for Kaggle environment to avoid potential issues
        ser_train_loader = DataLoader(
            ser_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        ser_test_loader = DataLoader(
            ser_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        sid_train_loader = DataLoader(
            sid_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        sid_test_loader = DataLoader(
            sid_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Created DataLoaders with {len(ser_train_dataset)} training samples and {len(ser_test_dataset)} testing samples")
        
        return {
            'ser_train': ser_train_loader,
            'ser_test': ser_test_loader,
            'sid_train': sid_train_loader,
            'sid_test': sid_test_loader
        }


# Example usage
if __name__ == "__main__":
    # Define paths
    ravdess_path = "/kaggle/input/ravdess-emotional-speech-audio"
    ser_save_path = "/kaggle/working/ser_probe.pth"
    sid_save_path = "/kaggle/working/sid_probe.pth"
    results_save_path = "/kaggle/working/ablation_results.png"
    
    # Create the ablation study object
    ablation_study = LayerwiseSignFlipAblation(
        model_name='microsoft/wavlm-base-plus',
        batch_size=8
    )
    
    try:
        # Check if the path exists
        if not os.path.exists(ravdess_path):
            raise FileNotFoundError(f"RAVDESS dataset path not found: {ravdess_path}")
            
        # Check if we can use pre-trained probes
        if os.path.exists(ser_save_path) and os.path.exists(sid_save_path):
            print(f"Found pre-trained probes. Loading from {ser_save_path} and {sid_save_path}...")
            # Create a new instance with the pre-trained probes
            ablation_study = LayerwiseSignFlipAblation(
                model_name='microsoft/wavlm-base-plus',
                ser_probe_path=ser_save_path,
                sid_probe_path=sid_save_path,
                batch_size=8
            )
            # Create only test dataloaders
            print("Creating RAVDESS test dataloaders...")
            dataloaders = ablation_study.create_ravdess_dataloaders(
                ravdess_path=ravdess_path,
                batch_size=8
            )
        else:
            # Create all dataloaders and train probes
            print("Creating RAVDESS dataloaders...")
            dataloaders = ablation_study.create_ravdess_dataloaders(
                ravdess_path=ravdess_path,
                batch_size=8
            )
            
            # Train probes on the RAVDESS dataset
            print("Training probes on RAVDESS dataset...")
            ablation_study.create_and_train_probes(
                ser_train_dataloader=dataloaders['ser_train'],
                sid_train_dataloader=dataloaders['sid_train'],
                num_epochs=3,  # Can increase for better results
                ser_save_path=ser_save_path,
                sid_save_path=sid_save_path
            )
        
        # Use the test datasets for ablation study
        print("Running ablation study on test datasets...")
        ablation_study.run_ablation_study(
            ser_dataloader=dataloaders['ser_test'],
            sid_dataloader=dataloaders['sid_test']
        )
        
    except FileNotFoundError:
        # Fall back to using dummy data if RAVDESS dataset is not available
        print("RAVDESS dataset not found, falling back to dummy data...")
        
        def load_ravdess_dataloaders(batch_size=16):
            """Load dummy dataloaders for demonstration."""
            from torch.utils.data import TensorDataset, DataLoader
            import torch
            
            # Create dummy datasets for demonstration
            
            # Dummy SER dataset (10 samples, 16000 audio samples each, 8 emotion classes)
            ser_inputs = torch.randn(10, 16000)
            ser_labels = torch.randint(0, 8, (10,))
            ser_dataset = TensorDataset(ser_inputs, ser_labels)
            ser_dataloader = DataLoader(
                ser_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                collate_fn=lambda batch: {
                    "input_values": torch.stack([item[0] for item in batch]),
                    "labels": torch.stack([item[1] for item in batch])
                }
            )
            
            # Dummy SID dataset (10 samples, 16000 audio samples each, 24 speaker classes)
            sid_inputs = torch.randn(10, 16000)
            sid_labels = torch.randint(0, 24, (10,))
            sid_dataset = TensorDataset(sid_inputs, sid_labels)
            sid_dataloader = DataLoader(
                sid_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                collate_fn=lambda batch: {
                    "input_values": torch.stack([item[0] for item in batch]),
                    "labels": torch.stack([item[1] for item in batch])
                }
            )
            
            return ser_dataloader, sid_dataloader
        
        # Load dummy dataloaders
        print("Loading dummy dataloaders...")
        ser_dataloader, sid_dataloader = load_ravdess_dataloaders(batch_size=8)
        
        # Run the ablation study with dummy data
        print("Running ablation study with dummy data...")
        ablation_study.run_ablation_study(ser_dataloader, sid_dataloader)
    
    # Visualize results
    print("Visualizing results...")
    ablation_study.visualize_results(save_path=results_save_path)
    
    # Get results summary
    summary = ablation_study.get_results_summary()
    print("\nResults Summary:")
    print(f"Maximum SER drop at layer {summary['max_ser_drop']['layer']} with drop {summary['max_ser_drop']['drop']:.4f}")
    print(f"Maximum SID drop at layer {summary['max_sid_drop']['layer']} with drop {summary['max_sid_drop']['drop']:.4f}")
    print(f"SER hypothesis confirmed: {summary['hypotheses']['ser_confirmed']}")
    print(f"SID hypothesis confirmed: {summary['hypotheses']['sid_confirmed']}")