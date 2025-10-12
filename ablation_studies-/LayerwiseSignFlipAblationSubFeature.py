"""
LayerwiseSignFlipAblation.py: Implementation of layer-wise sign-flipping ablation
for transformer-based speech models like WavLM.

This module provides a comprehensive class for performing sign-flipping ablation studies
across transformer layers to measure the impact on downstream speech tasks and on the
model's ability to represent low-level acoustic features.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Callable
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor
from torch.utils.data import DataLoader
import librosa
import re # Make sure re is imported

try:
    import parselmouth
except ImportError:
    print("Warning: parselmouth not found. Please run 'pip install parselmouth-praat'. Acoustic feature extraction will fail.")

# <<< NEW HELPER CLASS FOR ACOUSTIC FEATURE EXTRACTION >>>
class AcousticFeatureExtractor:
    """A helper class to extract various acoustic features from an audio file."""
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate

    def __call__(self, waveform: np.ndarray) -> Dict[str, float]:
        """
        Extracts all supported acoustic features from a raw waveform.

        Args:
            waveform (np.ndarray): The audio waveform.

        Returns:
            Dict[str, float]: A dictionary of extracted scalar feature values.
        """
        # Use parselmouth for pitch, jitter, HNR, and formants
        try:
            sound = parselmouth.Sound(waveform, sampling_frequency=self.sr)
            pitch = sound.to_pitch()
            # Calculate mean pitch, handling frames with no pitch
            pitch_values = pitch.selected_array['frequency']
            pitch_mean = np.nanmean(pitch_values[pitch_values != 0]) if np.any(pitch_values) else 0.0
            
            point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", pitch.floor, pitch.ceiling)
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100 # In %
            
            harmonicity = sound.to_harmonicity()
            # Calculate mean HNR, handling undefined frames
            hnr_values = harmonicity.values[harmonicity.values != -200]
            hnr = np.nanmean(hnr_values) if len(hnr_values) > 0 else 0.0

            formants = sound.to_formant_burg()
            f1_mean = np.nanmean([formants.get_value_at_time(1, parselmouth.FormantUnit.HERTZ, t) for t in formants.ts()]) if formants.nt > 0 else 0.0
            f2_mean = np.nanmean([formants.get_value_at_time(2, parselmouth.FormantUnit.HERTZ, t) for t in formants.ts()]) if formants.nt > 0 else 0.0
        
        except Exception as e:
            # Fallback if parselmouth fails
            # print(f"Parselmouth feature extraction failed: {e}. Using 0.0 as fallback.")
            pitch_mean, jitter, hnr, f1_mean, f2_mean = 0.0, 0.0, 0.0, 0.0, 0.0

        # Use librosa for other features
        loudness = librosa.feature.rms(y=waveform).mean()
        # Use onsets to get a more reliable tempo
        onsets = librosa.onset.onset_detect(y=waveform, sr=self.sr)
        speech_rate = librosa.beat.tempo(onset_envelope=librosa.onset.onset_strength(y=waveform, sr=self.sr), sr=self.sr)[0] if len(onsets) > 2 else 60.0 # fallback tempo
        zcr = librosa.feature.zero_crossing_rate(y=waveform).mean()
        mfccs = librosa.feature.mfcc(y=waveform, sr=self.sr, n_mfcc=13)
        mfcc_mean = mfccs.mean()

        return {
            # SER Features
            "Pitch": float(pitch_mean),
            "Loudness": float(loudness),
            "Speech Rate": float(speech_rate),
            "Jitter": float(jitter),
            "Harmonic-to-Noise Ratio": float(hnr),
            "Zero Crossing Rate": float(zcr),
            # SID Features (reusing some and adding new ones)
            "Average Spectral Shape (MFCCs)": float(mfcc_mean),
            "Vocal Tract Resonance (F1)": float(f1_mean),
            "Vocal Tract Resonance (F2)": float(f2_mean),
        }

class LayerwiseSignFlipAblation:
    def __init__(self, model_name: str = 'microsoft/wavlm-base-plus', **kwargs):
        self.model_name = model_name
        self.device = kwargs.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = kwargs.get('batch_size', 16)
        seed = kwargs.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Loading model: {model_name}")
        self.model = self._load_model(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.num_layers = len(self.model.encoder.layers)
        print(f"Model has {self.num_layers} layers")
        self.ser_probe = self._load_probe(kwargs.get('ser_probe_path'), "SER", 8)
        self.sid_probe = self._load_probe(kwargs.get('sid_probe_path'), "SID", 24)
        self.current_hook = None
        self.baseline_results = {}
        self.ablation_results = {}

        self.acoustic_features_list = [
            "Pitch", "Loudness", "Speech Rate", "Jitter",
            "Harmonic-to-Noise Ratio", "Zero Crossing Rate",
            "Average Spectral Shape (MFCCs)", "Vocal Tract Resonance (F1)",
            "Vocal Tract Resonance (F2)"
        ]
        self.acoustic_probes = {fname: self._load_probe(None, fname, 1) for fname in self.acoustic_features_list}
        self.acoustic_baseline_results = {}
        self.acoustic_ablation_results = {}

    def _load_model(self, model_name: str) -> torch.nn.Module:
        try:
            model = AutoModel.from_pretrained(model_name)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _load_probe(self, probe_path: Optional[str], probe_type: str, num_classes: int) -> torch.nn.Module:
        if probe_path is None:
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
        def hook(module, input_tensor, output_tensor):
            if isinstance(output_tensor, tuple):
                return (-output_tensor[0],) + output_tensor[1:]
            else:
                return -output_tensor
        return hook

    def register_hook(self, layer_idx: int):
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of bounds")
        self.remove_hook()
        try:
            module = self.model.encoder.layers[layer_idx].feed_forward.intermediate_act_fn
            self.current_hook = module.register_forward_hook(self.sign_flip_hook(layer_idx))
        except AttributeError:
            module = self.model.encoder.layers[layer_idx]
            self.current_hook = module.register_forward_hook(self.sign_flip_hook(layer_idx))

    def remove_hook(self):
        if self.current_hook:
            self.current_hook.remove()
            self.current_hook = None

    def run_evaluation(self, dataloader: DataLoader, probe_type: str = "SER") -> float:
        probe = self.ser_probe if probe_type == "SER" else self.sid_probe
        label_key = "labels" if probe_type == "SER" else "actor_labels"
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {probe_type}", leave=False):
                input_values = batch["input_values"].to(self.device)
                labels = batch[label_key].to(self.device)
                outputs = self.model(input_values=input_values)
                pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
                logits = probe(pooled_output)
                _, predicted = torch.max(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0
        print(f"{probe_type} Accuracy: {accuracy:.4f}")
        return accuracy
    
    def run_acoustic_evaluation(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs evaluation for all acoustic regression probes and returns their MSE."""
        mse_results = {fname: 0.0 for fname in self.acoustic_features_list}
        total_samples = 0
        criterion = torch.nn.MSELoss(reduction='sum')
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating Acoustic Probes", leave=False):
                input_values = batch["input_values"].to(self.device)
                outputs = self.model(input_values=input_values)
                pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
                total_samples += input_values.size(0)
                
                for fname in self.acoustic_features_list:
                    probe = self.acoustic_probes[fname]
                    probe.eval()
                    ground_truth = batch[fname].to(self.device).float().unsqueeze(1)
                    predictions = probe(pooled_output)
                    loss = criterion(predictions, ground_truth)
                    mse_results[fname] += loss.item()

        for fname in self.acoustic_features_list:
            mse_results[fname] /= total_samples if total_samples > 0 else 1
        return mse_results

    def run_baseline_evaluation(self, ser_dataloader: DataLoader, sid_dataloader: DataLoader, acoustic_dataloader: DataLoader):
        print("Running baseline evaluation...")
        self.remove_hook()
        
        # Classification tasks
        ser_accuracy = self.run_evaluation(ser_dataloader, "SER")
        sid_accuracy = self.run_evaluation(sid_dataloader, "SID")
        self.baseline_results = {"SER": ser_accuracy, "SID": sid_accuracy}

        print("Running baseline evaluation for acoustic features...")
        self.acoustic_baseline_results = self.run_acoustic_evaluation(acoustic_dataloader)
        print("Baseline Acoustic MSEs:", {k: f"{v:.4f}" for k, v in self.acoustic_baseline_results.items()})

    def run_ablation_study(self, ser_dataloader: DataLoader, sid_dataloader: DataLoader, acoustic_dataloader: DataLoader):
        if not self.baseline_results:
            self.run_baseline_evaluation(ser_dataloader, sid_dataloader, acoustic_dataloader)

        ser_drops, sid_drops = [], []
        acoustic_mse_increases = {fname: [] for fname in self.acoustic_features_list}

        for layer_idx in range(self.num_layers):
            print(f"\n--- Layer {layer_idx} Ablation ---")
            self.register_hook(layer_idx)
            
            # Classification ablation
            ser_acc = self.run_evaluation(ser_dataloader, "SER")
            sid_acc = self.run_evaluation(sid_dataloader, "SID")
            ser_drops.append(self.baseline_results["SER"] - ser_acc)
            sid_drops.append(self.baseline_results["SID"] - sid_acc)

            # Acoustic probes ablation
            ablated_mses = self.run_acoustic_evaluation(acoustic_dataloader)
            for fname in self.acoustic_features_list:
                baseline_mse = self.acoustic_baseline_results.get(fname, 0)
                ablated_mse = ablated_mses.get(fname, 0)
                increase = ablated_mse - baseline_mse
                acoustic_mse_increases[fname].append(increase)
            
            self.remove_hook()
            print(f"Layer {layer_idx} - SER Drop: {ser_drops[-1]:.4f}, SID Drop: {sid_drops[-1]:.4f}")

        self.ablation_results = {"SER": ser_drops, "SID": sid_drops}
        self.acoustic_ablation_results = acoustic_mse_increases
        return self.ablation_results

    def visualize_results(self, save_path: Optional[str] = None) -> None:
        if not self.ablation_results:
            raise ValueError("No classification ablation results to visualize. Run the ablation study first.")
        
        layer_indices = list(range(self.num_layers))
        ser_drops = self.ablation_results["SER"]
        sid_drops = self.ablation_results["SID"]
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True)
        fig.suptitle('Layer-wise Ablation: Classification Accuracy Drop', fontsize=18)

        axes[0].plot(layer_indices, ser_drops, 'b-o', label='SER Accuracy Drop')
        axes[0].set_ylabel('Accuracy Drop', fontsize=14)
        axes[0].set_title('SER Ablation', fontsize=16)
        axes[0].grid(True, alpha=0.5)
        axes[0].legend()

        axes[1].plot(layer_indices, sid_drops, 'r-s', label='SID Accuracy Drop')
        axes[1].set_xlabel('Layer Index', fontsize=14)
        axes[1].set_ylabel('Accuracy Drop', fontsize=14)
        axes[1].set_title('SID Ablation', fontsize=16)
        axes[1].grid(True, alpha=0.5)
        axes[1].legend()

        plt.xticks(layer_indices)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Classification visualization saved to {save_path}")
        plt.show()
    
    def visualize_acoustic_results(self, save_path: Optional[str] = None) -> None:
        """Visualizes the increase in MSE for each acoustic feature probe."""
        if not self.acoustic_ablation_results:
            raise ValueError("No acoustic ablation results to visualize. Run the study first.")
            
        ser_features = ["Pitch", "Loudness", "Speech Rate", "Jitter", "Harmonic-to-Noise Ratio", "Zero Crossing Rate"]
        sid_features = ["Average Spectral Shape (MFCCs)", "Vocal Tract Resonance (F1)", "Vocal Tract Resonance (F2)", "Pitch", "Jitter"]
        
        layer_indices = list(range(self.num_layers))

        # --- SER Features Plot ---
        num_ser_features = len(ser_features)
        fig1, axes1 = plt.subplots(nrows=num_ser_features, ncols=1, figsize=(12, 4 * num_ser_features), sharex=True)
        if num_ser_features == 1: axes1 = [axes1] # Ensure axes1 is always iterable
        fig1.suptitle('Ablation Impact on SER-related Acoustic Features (MSE Increase)', fontsize=18)
        for i, fname in enumerate(ser_features):
            mse_increases = self.acoustic_ablation_results.get(fname, [0]*self.num_layers)
            axes1[i].plot(layer_indices, mse_increases, marker='o', label=f'MSE Increase for {fname}')
            axes1[i].set_ylabel('MSE Increase')
            axes1[i].set_title(fname)
            axes1[i].grid(True, alpha=0.5)
        plt.xlabel('Layer Index')
        plt.xticks(layer_indices)
        fig1.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            ser_path = save_path.replace('.png', '_ser_features.png')
            plt.savefig(ser_path, dpi=300, bbox_inches='tight')
            print(f"SER acoustic visualization saved to {ser_path}")
        plt.show()

        # --- SID Features Plot ---
        num_sid_features = len(sid_features)
        fig2, axes2 = plt.subplots(nrows=num_sid_features, ncols=1, figsize=(12, 4 * num_sid_features), sharex=True)
        if num_sid_features == 1: axes2 = [axes2] # Ensure axes2 is always iterable
        fig2.suptitle('Ablation Impact on SID-related Acoustic Features (MSE Increase)', fontsize=18)
        for i, fname in enumerate(sid_features):
            mse_increases = self.acoustic_ablation_results.get(fname, [0]*self.num_layers)
            axes2[i].plot(layer_indices, mse_increases, marker='s', color='purple', label=f'MSE Increase for {fname}')
            axes2[i].set_ylabel('MSE Increase')
            axes2[i].set_title(fname)
            axes2[i].grid(True, alpha=0.5)
        plt.xlabel('Layer Index')
        plt.xticks(layer_indices)
        fig2.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            sid_path = save_path.replace('.png', '_sid_features.png')
            plt.savefig(sid_path, dpi=300, bbox_inches='tight')
            print(f"SID acoustic visualization saved to {sid_path}")
        plt.show()
    
    def create_and_train_acoustic_probes(self, train_dataloader: DataLoader, num_epochs: int = 10, learning_rate: float = 1e-3):
        """Trains linear regression probes for each acoustic feature."""
        print("Training acoustic feature regression probes...")
        criterion = torch.nn.MSELoss()
        
        for fname in self.acoustic_features_list:
            print(f"--- Training probe for: {fname} ---")
            probe = self.acoustic_probes[fname]
            optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
            
            for epoch in range(num_epochs):
                probe.train()
                total_loss = 0
                for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                    input_values = batch["input_values"].to(self.device)
                    ground_truth = batch[fname].to(self.device).float().unsqueeze(1)
                    
                    optimizer.zero_grad()
                    with torch.no_grad():
                        outputs = self.model(input_values=input_values)
                        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
                    
                    predictions = probe(pooled_output)
                    loss = criterion(predictions, ground_truth)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                print(f"Epoch {epoch+1} Loss for {fname}: {total_loss / len(train_dataloader):.6f}")
    
    def create_and_train_probes(self, ser_train_dataloader, sid_train_dataloader, num_epochs=5, learning_rate=1e-4, ser_save_path=None, sid_save_path=None):
        print("Training classification probes...")
        hidden_size = self.model.config.hidden_size
        
        # Train SER Probe
        ser_probe = self.ser_probe
        ser_probe.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(ser_probe.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            ser_probe.train()
            for batch in tqdm(ser_train_dataloader, desc=f"SER Train Epoch {epoch+1}", leave=False):
                 # ... training logic from original code ...
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    outputs = self.model(input_values)
                    pooled = torch.mean(outputs.last_hidden_state, dim=1)
                logits = ser_probe(pooled)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
        if ser_save_path: torch.save(ser_probe.state_dict(), ser_save_path)

        # Train SID Probe
        sid_probe = self.sid_probe
        sid_probe.to(self.device)
        optimizer = torch.optim.Adam(sid_probe.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            sid_probe.train()
            for batch in tqdm(sid_train_dataloader, desc=f"SID Train Epoch {epoch+1}", leave=False):
                # ... training logic from original code ...
                input_values = batch["input_values"].to(self.device)
                labels = batch["actor_labels"].to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    outputs = self.model(input_values)
                    pooled = torch.mean(outputs.last_hidden_state, dim=1)
                logits = sid_probe(pooled)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
        if sid_save_path: torch.save(sid_probe.state_dict(), sid_save_path)
        
    def create_ravdess_dataloaders(self, ravdess_path: str, **kwargs) -> Dict[str, DataLoader]:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        class RavdessDataset(torch.utils.data.Dataset):
            def __init__(self, metadata_df, feature_extractor_hf, acoustic_feature_extractor):
                self.metadata_df = metadata_df
                self.feature_extractor_hf = feature_extractor_hf
                self.acoustic_feature_extractor = acoustic_feature_extractor
                self.target_sr = 16000
                self.max_length = 160000

            def __len__(self):
                return len(self.metadata_df)

            def __getitem__(self, idx):
                row = self.metadata_df.iloc[idx]
                audio_file = row['file']
                try:
                    waveform, _ = librosa.load(audio_file, sr=self.target_sr, mono=True)
                except Exception as e:
                    print(f"Error loading {audio_file}: {e}. Skipping.")
                    return self.__getitem__((idx + 1) % len(self)) # Return next item on error

                if len(waveform) > self.max_length: waveform = waveform[:self.max_length]
                else: waveform = np.pad(waveform, (0, self.max_length - len(waveform)), 'constant')
                
                inputs_hf = self.feature_extractor_hf(waveform, sampling_rate=self.target_sr, return_tensors="pt")
                acoustic_features = self.acoustic_feature_extractor(waveform)
                
                item = {
                    "input_values": inputs_hf.input_values.squeeze(),
                    "labels": torch.tensor(row['emotion'], dtype=torch.long),
                    "actor_labels": torch.tensor(row['actor'], dtype=torch.long),
                }
                for fname, fval in acoustic_features.items():
                    item[fname] = torch.tensor(fval, dtype=torch.float32)
                return item

        print(f"Loading RAVDESS dataset from {ravdess_path}")
        feature_extractor_hf = AutoFeatureExtractor.from_pretrained(self.model_name)
        acoustic_feature_extractor = AcousticFeatureExtractor(sample_rate=16000)
        
        audio_files = [os.path.join(r, f) for r, d, files in os.walk(ravdess_path) for f in files if f.endswith('.wav')]
        
        # <<< FIX IS HERE: Robust filename parsing >>>
        metadata = []
        for file_path in audio_files:
            filename = os.path.basename(file_path)
            parts = re.match(r'(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)\.wav', filename)
            if parts:
                metadata.append({
                    'file': file_path,
                    'emotion': int(parts.group(3)) - 1,
                    'actor': int(parts.group(7)) - 1
                })
        
        if not metadata:
            raise ValueError("No valid RAVDESS files found. Check path and file naming.")

        df = pd.DataFrame(metadata)
        train_df, test_df = train_test_split(df, train_size=0.8, random_state=42, stratify=df[['emotion', 'actor']])

        train_dataset = RavdessDataset(train_df, feature_extractor_hf, acoustic_feature_extractor)
        test_dataset = RavdessDataset(test_df, feature_extractor_hf, acoustic_feature_extractor)
        
        batch_size = kwargs.get('batch_size', 16)
        
        acoustic_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        acoustic_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return {
            'ser_train': acoustic_train_loader, 'ser_test': acoustic_test_loader,
            'sid_train': acoustic_train_loader, 'sid_test': acoustic_test_loader,
            'acoustic_train': acoustic_train_loader, 'acoustic_test': acoustic_test_loader
        }

# Example usage
if __name__ == "__main__":
    ravdess_path = "/kaggle/input/ravdess-emotional-speech-audio"
    output_dir = "/kaggle/working/ablation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    ser_save_path = os.path.join(output_dir, "ser_probe.pth")
    sid_save_path = os.path.join(output_dir, "sid_probe.pth")
    cls_results_save_path = os.path.join(output_dir, "ablation_cls_results.png")
    acoustic_results_save_path = os.path.join(output_dir, "ablation_acoustic_results.png")

    ablation_study = LayerwiseSignFlipAblation(
        model_name='microsoft/wavlm-base-plus',
        batch_size=8
    )
    
    try:
        if not os.path.exists(ravdess_path) or not os.listdir(ravdess_path):
             raise FileNotFoundError
        dataloaders = ablation_study.create_ravdess_dataloaders(ravdess_path=ravdess_path, batch_size=8)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading dataset: {e}. Aborting.")
        exit()

    if not os.path.exists(ser_save_path) or not os.path.exists(sid_save_path):
        print("\n--- Training Classification Probes ---")
        ablation_study.create_and_train_probes(
            ser_train_dataloader=dataloaders['ser_train'],
            sid_train_dataloader=dataloaders['sid_train'],
            ser_save_path=ser_save_path,
            sid_save_path=sid_save_path
        )
    else:
        print("\n--- Loading existing classification probes ---")
        ablation_study.ser_probe.load_state_dict(torch.load(ser_save_path))
        ablation_study.sid_probe.load_state_dict(torch.load(sid_save_path))


    print("\n--- Training Acoustic Regression Probes ---")
    ablation_study.create_and_train_acoustic_probes(dataloaders['acoustic_train'], num_epochs=5)

    print("\n--- Running Full Ablation Study ---")
    ablation_study.run_ablation_study(
        ser_dataloader=dataloaders['ser_test'],
        sid_dataloader=dataloaders['sid_test'],
        acoustic_dataloader=dataloaders['acoustic_test']
    )

    print("\n--- Visualizing Results ---")
    ablation_study.visualize_results(save_path=cls_results_save_path)
    ablation_study.visualize_acoustic_results(save_path=acoustic_results_save_path)

    print("\n--- Ablation Study Complete ---")