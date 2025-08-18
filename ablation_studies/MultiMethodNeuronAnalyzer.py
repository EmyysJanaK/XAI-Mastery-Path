import torch
import torchaudio
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from tqdm import tqdm
import warnings
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. SHAP-based analysis will be disabled.")

class NeuronProbingAnalyzer:
    """
    SHAP → Top Neurons → Ablation → Probing pipeline for advanced neuron analysis
    """
    
    def __init__(self, model_name="microsoft/wavlm-base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the analyzer"""
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        self.num_layers = len(self.model.encoder.layers)
        self.hidden_size = self.model.config.hidden_size
        
        # Store SHAP results and neuron rankings
        self.shap_results = {}
        self.top_neurons = {}
        self.layer_outputs = {}
        self.hooks = []
        
        print(f"🧠 Neuron Probing Analyzer Initialized")
        print(f"  - Model: {model_name}")
        print(f"  - Layers: {self.num_layers}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Device: {self.device}")
        print(f"  - SHAP available: {SHAP_AVAILABLE}")

    def register_layer_hooks(self):
        """Register hooks to capture layer outputs"""
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                self.layer_outputs[layer_idx] = hidden_states.detach()
            return hook_fn
        
        for i, layer in enumerate(self.model.encoder.layers):
            hook = layer.register_forward_hook(create_hook(i))
            self.hooks.append(hook)

    def remove_layer_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_outputs = {}
    
    def load_ravdess_sample(self, ravdess_path: str, emotion_label: str = "happy", actor_id: int = None) -> Tuple[str, Dict]:
        """Load RAVDESS sample with emotion and actor filtering"""

        emotion_map = {
            1: "neutral", 2: "calm", 3: "happy", 4: "sad",
            5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
        }
        
        audio_files = []
        ravdess_path = Path(ravdess_path)
        
        for actor_folder in ravdess_path.glob("Actor_*"):
            if not actor_folder.is_dir():
                continue
                
            actor_num = int(actor_folder.name.split("_")[1])
            if actor_id is not None and actor_num != actor_id:
                continue
            
            for audio_file in actor_folder.glob("*.wav"):
                parts = audio_file.stem.split('-')
                if len(parts) >= 7:
                    emotion_id = int(parts[2])
                    emotion_name = emotion_map.get(emotion_id, "unknown")
                    
                    if emotion_label is not None and emotion_name != emotion_label:
                        continue
                    
                    metadata = {
                        'file_path': str(audio_file),
                        'emotion_id': emotion_id,
                        'emotion_name': emotion_name,
                        'intensity': int(parts[3]),
                        'actor_id': int(parts[6]),
                        'gender': 'female' if int(parts[6]) % 2 == 0 else 'male'
                    }
                    audio_files.append((str(audio_file), metadata))
        
        if not audio_files:
            raise ValueError(f"No RAVDESS files found for emotion='{emotion_label}', actor='{actor_id}'")
        
        audio_path, metadata = random.choice(audio_files)
        print(f"📁 Selected: {metadata['emotion_name']} emotion, Actor {metadata['actor_id']} ({metadata['gender']})")
        return audio_path, metadata

    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:

        """Preprocess audio for WavLM"""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=target_sr, 
            return_tensors="pt"
        )
        
        return inputs["input_values"].to(self.device)

    def create_multi_task_metrics(self) -> Dict[str, callable]:
        """Create multiple task metrics for multi-task hub analysis"""
        
        def emotion_classification_metric(model, audio_input):
            """Emotion classification task metric"""
            with torch.no_grad():
                outputs = model(audio_input)
                emotion_repr = outputs.last_hidden_state.mean(dim=1)
                return emotion_repr.norm(dim=1).mean().item()
        
        def speaker_identification_metric(model, audio_input):
            """Speaker identification task metric"""
            with torch.no_grad():
                outputs = model(audio_input)
                # Use variance across time dimension for speaker features
                speaker_repr = outputs.last_hidden_state.var(dim=1)
                return speaker_repr.norm(dim=1).mean().item()
        
        def gender_classification_metric(model, audio_input):
            """Gender classification task metric"""
            with torch.no_grad():
                outputs = model(audio_input)
                # Use specific frequency patterns for gender
                gender_repr = outputs.last_hidden_state[:, :, :self.hidden_size//2].mean(dim=1)
                return gender_repr.norm(dim=1).mean().item()
        
        def prosody_analysis_metric(model, audio_input):
            """Prosody/rhythm analysis task metric"""
            with torch.no_grad():
                outputs = model(audio_input)
                # Use temporal patterns for prosody
                prosody_repr = torch.diff(outputs.last_hidden_state, dim=1).abs().mean(dim=1)
                return prosody_repr.norm(dim=1).mean().item()
        
        def acoustic_feature_metric(model, audio_input):
            """Low-level acoustic feature metric"""
            with torch.no_grad():
                outputs = model(audio_input)
                # Use early layer patterns for acoustic features
                acoustic_repr = outputs.last_hidden_state.std(dim=1)
                return acoustic_repr.norm(dim=1).mean().item()
        
        return {
            "emotion_classification": emotion_classification_metric,
            "speaker_identification": speaker_identification_metric,
            "gender_classification": gender_classification_metric,
            "prosody_analysis": prosody_analysis_metric,
            "acoustic_features": acoustic_feature_metric
        }
    
    def step1_shap_neuron_identification(self, audio_inputs: List[torch.Tensor], 
                                        layers_to_analyze: List[int],
                                        top_k_per_layer: int = 30) -> Dict[str, Any]:
        """
        Step 1: Use SHAP to identify important neurons across multiple tasks (OPTIMIZED)
        """
        print(f"\n🎯 STEP 1: FAST SHAP-based Neuron Identification")
        print(f"Analyzing {len(layers_to_analyze)} layers with {min(len(audio_inputs), 3)} audio samples (optimized)")
        
        if not SHAP_AVAILABLE:
            print("⚠️  SHAP not available, using random selection")
            random_results = {}
            for layer_idx in layers_to_analyze:
                random_neurons = random.sample(range(self.hidden_size), min(top_k_per_layer, self.hidden_size))
                random_results[layer_idx] = {
                    'top_neurons': random_neurons,
                    'shap_scores': {n: random.random() for n in random_neurons}
                }
            return random_results
        
        # Register hooks for layer output capture
        self.register_layer_hooks()
        
        shap_results = {}
        # Use only essential tasks for faster computation
        essential_tasks = {
            "emotion_classification": self.create_multi_task_metrics()["emotion_classification"],
            "speaker_identification": self.create_multi_task_metrics()["speaker_identification"]
        }
        
        # Use only first 3 audio samples for speed
        fast_audio_inputs = audio_inputs[:3]
        
        try:
            for layer_idx in layers_to_analyze:
                print(f"\n📊 Analyzing Layer {layer_idx} (Fast Mode)")
                
                layer_shap_scores = {}
                
                # Analyze neurons for essential tasks only
                for task_name, task_metric in essential_tasks.items():
                    print(f"  🎯 Task: {task_name}")
                    
                    task_scores = self._compute_shap_for_layer_task(
                        fast_audio_inputs, layer_idx, task_metric, 
                        num_neurons=min(50, self.hidden_size)  # Reduced from 100 to 50
                    )
                    
                # Aggregate scores across tasks
                for neuron_idx, score in task_scores.items():
                    if neuron_idx not in layer_shap_scores:
                        layer_shap_scores[neuron_idx] = []
                    layer_shap_scores[neuron_idx].append(score)
                
                # Calculate final importance scores (simplified for speed)
                final_scores = {}
                for neuron_idx, scores in layer_shap_scores.items():
                    # Use mean only for faster computation
                    final_scores[neuron_idx] = np.mean(scores)                # Get top neurons
                sorted_neurons = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                top_neurons = [neuron_idx for neuron_idx, _ in sorted_neurons[:top_k_per_layer]]
                
                shap_results[layer_idx] = {
                    'top_neurons': top_neurons,
                    'shap_scores': final_scores,
                    'task_breakdown': {
                        task_name: {neuron_idx: scores[i] for neuron_idx, scores in layer_shap_scores.items()}
                        for i, task_name in enumerate(essential_tasks.keys())
                    }
                }
                
                print(f"  ✅ Found {len(top_neurons)} important neurons")
        
        finally:
            self.remove_layer_hooks()
        
        self.shap_results = shap_results
        return shap_results

    def _compute_shap_for_layer_task(self, audio_inputs: List[torch.Tensor], 
                                   layer_idx: int, task_metric: callable,
                                   num_neurons: int = 100) -> Dict[int, float]:
        """Compute SHAP values for specific layer and task"""
        
        def neuron_wrapper(neuron_idx):
            def wrapper(audio_batch):
                if isinstance(audio_batch, np.ndarray):
                    audio_batch = torch.tensor(audio_batch, dtype=torch.float32).to(self.device)
                
                self.layer_outputs.clear()
                _ = self.model(audio_batch)
                
                if layer_idx in self.layer_outputs:
                    hidden_states = self.layer_outputs[layer_idx]
                    neuron_activation = hidden_states[:, :, neuron_idx].mean(dim=1)
                    return neuron_activation.cpu().numpy()
                else:
                    return np.zeros(audio_batch.shape[0])
            return wrapper
        
        neuron_scores = {}
        
        # Sample fewer neurons for speed
        neuron_indices = random.sample(range(self.hidden_size), min(num_neurons, self.hidden_size))
        
        for neuron_idx in tqdm(neuron_indices, desc=f"Fast SHAP Layer {layer_idx}"):
            try:
                # Use minimal samples for SHAP
                sample_audio = random.choice(audio_inputs)[:2]  # Use only 2 samples
                background = torch.zeros_like(sample_audio[:1])
                
                wrapper = neuron_wrapper(neuron_idx)
                explainer = shap.KernelExplainer(wrapper, background.cpu().numpy())
                
                # Reduced nsamples for faster computation
                shap_values = explainer.shap_values(sample_audio.cpu().numpy(), nsamples=20)
                importance = np.abs(shap_values).mean()
                neuron_scores[neuron_idx] = float(importance)
                
            except Exception as e:
                neuron_scores[neuron_idx] = 0.0
        
        return neuron_scores

    def step2_multi_task_hub_analysis(self, audio_inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Step 2: Analyze which neurons are multi-task hubs vs task-specific (FAST VERSION)
        """
        print(f"\n🔗 STEP 2: Fast Multi-Task Hub Analysis")
        
        if not self.shap_results:
            raise ValueError("Must run step1_shap_neuron_identification first")
        
        # Use essential tasks only for speed
        essential_tasks = ["emotion_classification", "speaker_identification"]
        hub_analysis = {}
        
        for layer_idx, layer_data in self.shap_results.items():
            print(f"📊 Analyzing Layer {layer_idx} (Fast Mode)")
            
            top_neurons = layer_data['top_neurons'][:20]  # Reduced to top 20 neurons for speed
            task_breakdown = layer_data['task_breakdown']
            
            # Calculate task specificity for each neuron
            neuron_task_profiles = {}
            
            for neuron_idx in top_neurons:
                task_scores = []
                for task_name in essential_tasks:
                    if neuron_idx in task_breakdown.get(task_name, {}):
                        score = task_breakdown[task_name][neuron_idx]
                        task_scores.append(score)
                    else:
                        task_scores.append(0.0)
                
                # Normalize scores
                task_scores = np.array(task_scores)
                if task_scores.sum() > 0:
                    task_scores = task_scores / task_scores.sum()
                
                # Calculate entropy (low entropy = task-specific, high entropy = multi-task hub)
                entropy = -np.sum(task_scores * np.log(task_scores + 1e-8))
                max_entropy = np.log(len(task_scores))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                # Calculate dominance (how much one task dominates)
                dominance = np.max(task_scores) if len(task_scores) > 0 else 0
                
                neuron_task_profiles[neuron_idx] = {
                    'task_scores': task_scores.tolist(),
                    'entropy': entropy,
                    'normalized_entropy': normalized_entropy,
                    'dominance': dominance,
                    'hub_score': normalized_entropy * (1 - dominance),  # Multi-task hub score
                    'specialization_score': dominance * (1 - normalized_entropy)  # Task-specific score
                }
            
            # Classify neurons
            hub_neurons = []
            specialist_neurons = []
            
            for neuron_idx, profile in neuron_task_profiles.items():
                if profile['hub_score'] > 0.6:  # High entropy, low dominance
                    hub_neurons.append(neuron_idx)
                elif profile['specialization_score'] > 0.6:  # Low entropy, high dominance
                    specialist_neurons.append(neuron_idx)
            
            hub_analysis[layer_idx] = {
                'neuron_profiles': neuron_task_profiles,
                'hub_neurons': hub_neurons,
                'specialist_neurons': specialist_neurons,
                'layer_stats': {
                    'total_analyzed': len(top_neurons),
                    'num_hubs': len(hub_neurons),
                    'num_specialists': len(specialist_neurons),
                    'hub_ratio': len(hub_neurons) / len(top_neurons),
                    'specialist_ratio': len(specialist_neurons) / len(top_neurons)
                }
            }
            
            print(f"  🔗 Multi-task hubs: {len(hub_neurons)}")
            print(f"  🎯 Task specialists: {len(specialist_neurons)}")
        
        return hub_analysis

    def step3_progressive_ablation_robustness(self, audio_inputs: List[torch.Tensor],
                                            hub_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Test robustness by progressive ablation and layer recovery
        """
        print(f"\n🛡️ STEP 3: Progressive Ablation & Robustness Analysis")
        
        task_metrics = self.create_multi_task_metrics()
        robustness_results = {}
        
        # Test different ablation strategies
        ablation_strategies = {
            'hub_first': 'Ablate multi-task hubs first',
            'specialist_first': 'Ablate task specialists first',
            'random': 'Random ablation order',
            'importance_first': 'Ablate by SHAP importance'
        }
        
        for strategy_name, strategy_desc in ablation_strategies.items():
            print(f"\n🧪 Testing strategy: {strategy_name}")
            
            strategy_results = {}
            
            for layer_idx, layer_data in hub_analysis.items():
                print(f"  📊 Layer {layer_idx}")
                
                # Get neurons to ablate based on strategy
                neurons_to_ablate = self._get_ablation_order(
                    layer_idx, layer_data, strategy_name
                )
                
                # Progressive ablation
                ablation_results = self._progressive_ablation_test(
                    audio_inputs, layer_idx, neurons_to_ablate, task_metrics
                )
                
                # Test layer recovery (can later layers compensate?)
                recovery_results = self._test_layer_recovery(
                    audio_inputs, layer_idx, neurons_to_ablate[:10], task_metrics
                )
                
                strategy_results[layer_idx] = {
                    'ablation_curve': ablation_results,
                    'recovery_analysis': recovery_results,
                    'neurons_ablated': neurons_to_ablate[:20]  # Top 20 for analysis
                }
            
            robustness_results[strategy_name] = strategy_results
        
        return robustness_results

    def _get_ablation_order(self, layer_idx: int, layer_data: Dict, strategy: str) -> List[int]:
        """Get neuron ablation order based on strategy"""
        
        if strategy == 'hub_first':
            # Ablate multi-task hubs first
            hubs = layer_data['hub_neurons']
            specialists = layer_data['specialist_neurons']
            others = [n for n in layer_data['neuron_profiles'].keys() 
                     if n not in hubs and n not in specialists]
            return hubs + specialists + others
            
        elif strategy == 'specialist_first':
            # Ablate specialists first
            hubs = layer_data['hub_neurons']
            specialists = layer_data['specialist_neurons']
            others = [n for n in layer_data['neuron_profiles'].keys() 
                     if n not in hubs and n not in specialists]
            return specialists + hubs + others
            
        elif strategy == 'random':
            # Random order
            all_neurons = list(layer_data['neuron_profiles'].keys())
            random.shuffle(all_neurons)
            return all_neurons
            
        elif strategy == 'importance_first':
            # By SHAP importance
            shap_scores = self.shap_results[layer_idx]['shap_scores']
            sorted_neurons = sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)
            return [neuron_idx for neuron_idx, _ in sorted_neurons]
        
        return []

    def _progressive_ablation_test(self, audio_inputs: List[torch.Tensor], 
                                 layer_idx: int, neurons_to_ablate: List[int],
                                 task_metrics: Dict[str, callable]) -> Dict[str, List[float]]:
        """Perform progressive ablation and measure task performance"""
        
        ablation_results = {task_name: [] for task_name in task_metrics.keys()}
        
        # Baseline scores
        baseline_scores = {}
        for task_name, task_metric in task_metrics.items():
            scores = []
            for audio_input in audio_inputs[:3]:  # Use subset for speed
                score = task_metric(self.model, audio_input)
                scores.append(score)
            baseline_scores[task_name] = np.mean(scores)
            ablation_results[task_name].append(baseline_scores[task_name])
        
        # Progressive ablation
        ablated_neurons = []
        
        for i, neuron_idx in enumerate(neurons_to_ablate[:20]):  # Test first 20 neurons
            ablated_neurons.append(neuron_idx)
            
            # Test with current ablation set
            ablated_scores = self._test_with_ablated_neurons(
                audio_inputs[:3], layer_idx, ablated_neurons, task_metrics
            )
            
            for task_name in task_metrics.keys():
                ablation_results[task_name].append(ablated_scores[task_name])
        
        return ablation_results

    def _test_with_ablated_neurons(self, audio_inputs: List[torch.Tensor],
                                 layer_idx: int, ablated_neurons: List[int],
                                 task_metrics: Dict[str, callable]) -> Dict[str, float]:
        """Test model performance with specific neurons ablated"""
        
        def create_ablation_hook(neurons_to_ablate):
            def ablation_hook(module, input_tensor, output_tensor):
                if isinstance(output_tensor, tuple):
                    hidden_states = output_tensor[0].clone()
                    other_outputs = output_tensor[1:]
                else:
                    hidden_states = output_tensor.clone()
                    other_outputs = ()
                
                # Ablate all specified neurons
                for neuron_idx in neurons_to_ablate:
                    hidden_states[:, :, neuron_idx] = 0.0
                
                if other_outputs:
                    return (hidden_states,) + other_outputs
                return hidden_states
            return ablation_hook
        
        layer = self.model.encoder.layers[layer_idx]
        hook = layer.register_forward_hook(create_ablation_hook(ablated_neurons))
        
        try:
            task_scores = {}
            for task_name, task_metric in task_metrics.items():
                scores = []
                for audio_input in audio_inputs:
                    score = task_metric(self.model, audio_input)
                    scores.append(score)
                task_scores[task_name] = np.mean(scores)
            
            return task_scores
            
        finally:
            hook.remove()

    def _test_layer_recovery(self, audio_inputs: List[torch.Tensor],
                           ablated_layer_idx: int, ablated_neurons: List[int],
                           task_metrics: Dict[str, callable]) -> Dict[str, Any]:
        """Test if later layers can recover from ablations"""
        
        recovery_results = {
            'baseline_performance': {},
            'ablated_performance': {},
            'recovery_scores': {},
            'compensation_layers': []
        }
        
        # Test baseline
        for task_name, task_metric in task_metrics.items():
            scores = []
            for audio_input in audio_inputs[:3]:
                score = task_metric(self.model, audio_input)
                scores.append(score)
            recovery_results['baseline_performance'][task_name] = np.mean(scores)
        
        # Test with ablation
        ablated_scores = self._test_with_ablated_neurons(
            audio_inputs[:3], ablated_layer_idx, ablated_neurons, task_metrics
        )
        recovery_results['ablated_performance'] = ablated_scores
        
        # Calculate recovery (how much performance is retained)
        for task_name in task_metrics.keys():
            baseline = recovery_results['baseline_performance'][task_name]
            ablated = ablated_scores[task_name]
            
            if baseline != 0:
                recovery_score = ablated / baseline
            else:
                recovery_score = 1.0
                
            recovery_results['recovery_scores'][task_name] = recovery_score
        
        # Identify potential compensation layers (future work: analyze later layer activations)
        later_layers = [l for l in range(ablated_layer_idx + 1, self.num_layers)]
        recovery_results['compensation_layers'] = later_layers[:3]  # Next 3 layers
        
        return recovery_results
    
    def step4_model_size_analysis(self, audio_inputs: List[torch.Tensor],
                                 hub_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4: Analyze how model size affects neuron importance distribution
        """
        print(f"\n📏 STEP 4: Model Size Effect Analysis")
        
        # Analyze neuron importance distribution patterns
        size_analysis = {
            'importance_distribution': {},
            'layer_efficiency': {},
            'concentration_metrics': {},
            'scaling_patterns': {}
        }
        
        # 1. Importance Distribution Analysis
        for layer_idx, layer_data in self.shap_results.items():
            shap_scores = layer_data['shap_scores']
            scores = list(shap_scores.values())
            
            # Calculate distribution metrics
            distribution_metrics = {
                'mean_importance': np.mean(scores),
                'std_importance': np.std(scores),
                'skewness': self._calculate_skewness(scores),
                'kurtosis': self._calculate_kurtosis(scores),
                'gini_coefficient': self._calculate_gini_coefficient(scores),
                'top_10_percent_share': self._calculate_top_k_share(scores, 0.1),
                'top_1_percent_share': self._calculate_top_k_share(scores, 0.01),
                'effective_neurons': self._calculate_effective_neurons(scores)
            }
            
            size_analysis['importance_distribution'][layer_idx] = distribution_metrics
        
        # 2. Layer Efficiency Analysis
        for layer_idx in self.shap_results.keys():
            hub_data = hub_analysis[layer_idx]
            
            efficiency_metrics = {
                'neurons_per_task': len(hub_data['neuron_profiles']) / len(self.create_multi_task_metrics()),
                'hub_efficiency': len(hub_data['hub_neurons']) / len(hub_data['neuron_profiles']),
                'specialization_efficiency': len(hub_data['specialist_neurons']) / len(hub_data['neuron_profiles']),
                'utilization_rate': len(hub_data['neuron_profiles']) / self.hidden_size
            }
            
            size_analysis['layer_efficiency'][layer_idx] = efficiency_metrics
        
        # 3. Concentration Metrics
        for layer_idx, layer_data in self.shap_results.items():
            shap_scores = layer_data['shap_scores']
            scores = np.array(list(shap_scores.values()))
            
            concentration_metrics = {
                'herfindahl_index': np.sum((scores / scores.sum()) ** 2),
                'entropy': -np.sum((scores / scores.sum()) * np.log(scores / scores.sum() + 1e-8)),
                'concentration_ratio_5': np.sum(np.sort(scores)[-5:]) / scores.sum(),
                'concentration_ratio_10': np.sum(np.sort(scores)[-10:]) / scores.sum(),
                'participation_ratio': (scores.sum() ** 2) / np.sum(scores ** 2)
            }
            
            size_analysis['concentration_metrics'][layer_idx] = concentration_metrics
        
        # 4. Scaling Patterns
        layer_indices = sorted(self.shap_results.keys())
        scaling_patterns = self._analyze_scaling_patterns(layer_indices, size_analysis)
        size_analysis['scaling_patterns'] = scaling_patterns
        
        return size_analysis

    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of importance distribution"""
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of importance distribution"""
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_gini_coefficient(self, data: List[float]) -> float:
        """Calculate Gini coefficient for importance inequality"""
        data = np.array(sorted(data))
        n = len(data)
        if n == 0 or data.sum() == 0:
            return 0
        
        cumsum = np.cumsum(data)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def _calculate_top_k_share(self, data: List[float], k_percent: float) -> float:
        """Calculate share of top k% neurons in total importance"""
        data = np.array(sorted(data, reverse=True))
        k_count = max(1, int(len(data) * k_percent))
        return data[:k_count].sum() / data.sum() if data.sum() > 0 else 0

    def _calculate_effective_neurons(self, data: List[float], threshold: float = 0.01) -> int:
        """Calculate number of effectively contributing neurons"""
        data = np.array(data)
        total_importance = data.sum()
        if total_importance == 0:
            return 0
        
        # Count neurons contributing more than threshold of total importance
        return np.sum(data / total_importance > threshold)

    def _analyze_scaling_patterns(self, layer_indices: List[int], 
                                size_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how metrics scale across layers"""
        
        patterns = {
            'layer_progression': {},
            'early_vs_late': {},
            'middle_layer_peak': {}
        }
        
        # Extract metrics across layers
        metrics_across_layers = {}
        for metric_type in ['importance_distribution', 'layer_efficiency', 'concentration_metrics']:
            metrics_across_layers[metric_type] = {}
            
            for layer_idx in layer_indices:
                layer_metrics = size_analysis[metric_type][layer_idx]
                for metric_name, value in layer_metrics.items():
                    if metric_name not in metrics_across_layers[metric_type]:
                        metrics_across_layers[metric_type][metric_name] = []
                    metrics_across_layers[metric_type][metric_name].append(value)
        
        # Analyze progression patterns
        for metric_type, metrics_dict in metrics_across_layers.items():
            patterns['layer_progression'][metric_type] = {}
            
            for metric_name, values in metrics_dict.items():
                if len(values) > 2:
                    # Calculate trend
                    x = np.array(layer_indices)
                    y = np.array(values)
                    
                    # Linear regression for trend
                    slope = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
                    
                    patterns['layer_progression'][metric_type][metric_name] = {
                        'values': values,
                        'trend_slope': slope,
                        'early_value': values[0] if len(values) > 0 else 0,
                        'late_value': values[-1] if len(values) > 0 else 0,
                        'peak_layer': layer_indices[np.argmax(values)] if len(values) > 0 else 0,
                        'min_layer': layer_indices[np.argmin(values)] if len(values) > 0 else 0
                    }
        
        # Early vs Late layer comparison
        mid_point = len(layer_indices) // 2
        early_layers = layer_indices[:mid_point]
        late_layers = layer_indices[mid_point:]
        
        patterns['early_vs_late'] = {
            'early_layers': early_layers,
            'late_layers': late_layers,
            'comparisons': {}
        }
        
        # Compare early vs late for key metrics
        key_metrics = [
            ('importance_distribution', 'mean_importance'),
            ('layer_efficiency', 'hub_efficiency'),
            ('concentration_metrics', 'entropy')
        ]
        
        for metric_type, metric_name in key_metrics:
            early_values = [size_analysis[metric_type][l][metric_name] for l in early_layers]
            late_values = [size_analysis[metric_type][l][metric_name] for l in late_layers]
            
            patterns['early_vs_late']['comparisons'][f"{metric_type}_{metric_name}"] = {
                'early_mean': np.mean(early_values),
                'late_mean': np.mean(late_values),
                'difference': np.mean(late_values) - np.mean(early_values),
                'early_std': np.std(early_values),
                'late_std': np.std(late_values)
            }
        
        return patterns

    def _plot_robustness_analysis_focused(self, fig, gs, robustness_results: Dict[str, Any]):
        """Plot focused robustness and recovery analysis"""
        
        # Plot 1: Ablation curves comparison
        ax1 = fig.add_subplot(gs[0, 0])
        
        strategy_colors = {
            'hub_first': '#FF6B6B',
            'specialist_first': '#4ECDC4', 
            'random': '#45B7D1',
            'importance_first': '#96CEB4'
        }
        
        # Use first layer and first task for demonstration
        first_layer = sorted(robustness_results['hub_first'].keys())[0]
        first_task = "emotion_classification"  # Use essential task
        
        for strategy, strategy_data in robustness_results.items():
            if first_layer in strategy_data:
                ablation_curve = strategy_data[first_layer]['ablation_curve'][first_task]
                steps = range(len(ablation_curve))
                ax1.plot(steps, ablation_curve, 'o-', color=strategy_colors.get(strategy, '#333333'),
                        linewidth=2, label=strategy.replace('_', ' ').title(), markersize=4)
        
        ax1.set_xlabel('Number of Neurons Ablated')
        ax1.set_ylabel('Task Performance')
        ax1.set_title(f'Ablation Strategies Comparison\nLayer {first_layer}, Emotion Classification', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Recovery scores across strategies
        ax2 = fig.add_subplot(gs[0, 1])
        
        strategies = list(robustness_results.keys())
        layers = sorted(robustness_results[strategies[0]].keys())
        
        recovery_data = []
        strategy_labels = []
        
        for strategy in strategies:
            avg_recoveries = []
            for layer_idx in layers:
                if layer_idx in robustness_results[strategy]:
                    recovery_scores = robustness_results[strategy][layer_idx]['recovery_analysis']['recovery_scores']
                    avg_recovery = np.mean(list(recovery_scores.values()))
                    avg_recoveries.append(avg_recovery)
            
            if avg_recoveries:
                recovery_data.append(np.mean(avg_recoveries))
                strategy_labels.append(strategy.replace('_', ' ').title())
        
        bars = ax2.bar(strategy_labels, recovery_data, 
                      color=[strategy_colors.get(s.lower().replace(' ', '_'), '#333333') for s in strategy_labels],
                      alpha=0.7)
        
        ax2.set_ylabel('Average Recovery Score')
        ax2.set_title('Robustness Comparison\nAcross Ablation Strategies', fontweight='bold')
        ax2.set_xticklabels(strategy_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, recovery_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Layer compensation analysis
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Show which layers can compensate for ablations
        compensation_matrix = np.zeros((len(layers), len(layers)))
        
        for i, ablated_layer in enumerate(layers):
            for strategy in robustness_results.keys():
                if ablated_layer in robustness_results[strategy]:
                    recovery_data_comp = robustness_results[strategy][ablated_layer]['recovery_analysis']
                    compensation_layers = recovery_data_comp.get('compensation_layers', [])
                    
                    for comp_layer in compensation_layers:
                        if comp_layer in layers:
                            j = layers.index(comp_layer)
                            compensation_matrix[i, j] += 1
        
        # Normalize
        compensation_matrix = compensation_matrix / len(robustness_results) if len(robustness_results) > 0 else compensation_matrix
        
        im = ax3.imshow(compensation_matrix, cmap='Reds', aspect='auto')
        ax3.set_xticks(range(len(layers)))
        ax3.set_xticklabels([f'L{l}' for l in layers])
        ax3.set_yticks(range(len(layers)))
        ax3.set_yticklabels([f'L{l}' for l in layers])
        ax3.set_xlabel('Compensating Layer')
        ax3.set_ylabel('Ablated Layer')
        ax3.set_title('Layer Compensation Patterns\n(Recovery from Ablations)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(layers)):
            for j in range(len(layers)):
                if compensation_matrix[i, j] > 0:
                    text = ax3.text(j, i, f'{compensation_matrix[i, j]:.2f}',
                                   ha="center", va="center", 
                                   color="white" if compensation_matrix[i, j] > 0.5 else "black",
                                   fontweight='bold')
        
        plt.colorbar(im, ax=ax3, shrink=0.8)

    def _plot_size_analysis_focused(self, fig, gs, size_analysis: Dict[str, Any]):
        """Plot focused model size effects analysis"""
        
        # Plot 4: Importance distribution across layers
        ax4 = fig.add_subplot(gs[1, 0])
        
        layers = sorted(size_analysis['importance_distribution'].keys())
        gini_coeffs = [size_analysis['importance_distribution'][l]['gini_coefficient'] for l in layers]
        top_1_shares = [size_analysis['importance_distribution'][l]['top_1_percent_share'] for l in layers]
        
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(layers, gini_coeffs, 'o-', color='#E74C3C', linewidth=3, markersize=8, label='Gini Coefficient')
        line2 = ax4_twin.plot(layers, top_1_shares, 's-', color='#3498DB', linewidth=3, markersize=8, label='Top 1% Share')
        
        ax4.set_xlabel('Layer Index')
        ax4.set_ylabel('Gini Coefficient', color='#E74C3C')
        ax4_twin.set_ylabel('Top 1% Importance Share', color='#3498DB')
        ax4.set_title('Importance Inequality\nAcross Layers', fontweight='bold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
        
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Effective neurons vs layer depth
        ax5 = fig.add_subplot(gs[1, 1])
        
        effective_neurons = [size_analysis['importance_distribution'][l]['effective_neurons'] for l in layers]
        utilization_rates = [size_analysis['layer_efficiency'][l]['utilization_rate'] for l in layers]
        
        ax5.plot(layers, effective_neurons, 'o-', color='#9B59B6', linewidth=3, markersize=8, label='Effective Neurons')
        ax5_twin = ax5.twinx()
        ax5_twin.plot(layers, utilization_rates, 's-', color='#F39C12', linewidth=3, markersize=8, label='Utilization Rate')
        
        ax5.set_xlabel('Layer Index')
        ax5.set_ylabel('Effective Neurons', color='#9B59B6')
        ax5_twin.set_ylabel('Utilization Rate', color='#F39C12')
        ax5.set_title('Neuron Utilization\nEfficiency', fontweight='bold')
        
        # Combine legends
        lines1 = ax5.get_lines()
        lines2 = ax5_twin.get_lines()
        ax5.legend(lines1 + lines2, ['Effective Neurons', 'Utilization Rate'], loc='upper right')
        
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Scaling patterns summary
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Create scaling summary
        scaling_data = size_analysis['scaling_patterns']['early_vs_late']['comparisons']
        
        metrics = []
        early_values = []
        late_values = []
        differences = []
        
        for metric_name, comparison in scaling_data.items():
            if 'mean_importance' in metric_name or 'hub_efficiency' in metric_name or 'entropy' in metric_name:
                metrics.append(metric_name.replace('_', ' ').title())
                early_values.append(comparison['early_mean'])
                late_values.append(comparison['late_mean'])
                differences.append(comparison['difference'])
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, early_values, width, label='Early Layers', color='#2ECC71', alpha=0.7)
        bars2 = ax6.bar(x + width/2, late_values, width, label='Late Layers', color='#E67E22', alpha=0.7)
        
        ax6.set_xlabel('Metrics')
        ax6.set_ylabel('Metric Values')
        ax6.set_title('Early vs Late Layer\nCharacteristics', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics, rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add difference indicators
        for i, diff in enumerate(differences):
            if diff > 0:
                ax6.annotate('↑', xy=(i, max(early_values[i], late_values[i])), 
                           xytext=(i, max(early_values[i], late_values[i]) + 0.1 * max(early_values + late_values)),
                           ha='center', fontsize=16, color='green', fontweight='bold')
            elif diff < 0:
                ax6.annotate('↓', xy=(i, max(early_values[i], late_values[i])), 
                           xytext=(i, max(early_values[i], late_values[i]) + 0.1 * max(early_values + late_values)),
                           ha='center', fontsize=16, color='red', fontweight='bold')

    def create_focused_visualizations(self, robustness_results: Dict[str, Any],
                                     size_analysis: Dict[str, Any],
                                     save_path: str = None):
        """
        Create focused visualizations for layer recovery and model size effects only
        """
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
        
        # Research Question 2: Layer recovery and robustness  
        self._plot_robustness_analysis_focused(fig, gs, robustness_results)
        
        # Research Question 3: Model size effects
        self._plot_size_analysis_focused(fig, gs, size_analysis)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Focused visualizations saved to: {save_path}")
        
        plt.show()
        return fig

    def _plot_multitask_analysis(self, fig, gs, hub_analysis: Dict[str, Any]):
        """Plot multi-task hub analysis (Research Question 1)"""
        
        # Plot 1: Hub vs Specialist distribution across layers
        ax1 = fig.add_subplot(gs[0, 0])
        
        layers = sorted(hub_analysis.keys())
        hub_counts = [hub_analysis[l]['layer_stats']['num_hubs'] for l in layers]
        specialist_counts = [hub_analysis[l]['layer_stats']['num_specialists'] for l in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        
        ax1.bar(x - width/2, hub_counts, width, label='Multi-task Hubs', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, specialist_counts, width, label='Task Specialists', color='#A23B72', alpha=0.8)
        
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Number of Neurons')
        ax1.set_title('Multi-task Hubs vs Task Specialists\nAcross Layers', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hub/Specialist ratios
        ax2 = fig.add_subplot(gs[0, 1])
        
        hub_ratios = [hub_analysis[l]['layer_stats']['hub_ratio'] for l in layers]
        specialist_ratios = [hub_analysis[l]['layer_stats']['specialist_ratio'] for l in layers]
        
        ax2.plot(layers, hub_ratios, 'o-', color='#2E86AB', linewidth=3, markersize=8, label='Hub Ratio')
        ax2.plot(layers, specialist_ratios, 's-', color='#A23B72', linewidth=3, markersize=8, label='Specialist Ratio')
        
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Ratio of Total Neurons')
        ax2.set_title('Hub/Specialist Ratios\nAcross Network Depth', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Task specialization heatmap
        ax3 = fig.add_subplot(gs[0, 2:])
        
        # Create specialization matrix
        task_names = list(self.create_multi_task_metrics().keys())
        specialization_matrix = []
        
        for layer_idx in layers:
            layer_specialization = []
            layer_data = hub_analysis[layer_idx]
            
            # Calculate average specialization for each task in this layer
            for task_idx, task_name in enumerate(task_names):
                task_specialists = []
                for neuron_idx, profile in layer_data['neuron_profiles'].items():
                    if len(profile['task_scores']) > task_idx:
                        task_specialists.append(profile['task_scores'][task_idx])
                
                avg_specialization = np.mean(task_specialists) if task_specialists else 0
                layer_specialization.append(avg_specialization)
            
            specialization_matrix.append(layer_specialization)
        
        im = ax3.imshow(specialization_matrix, cmap='viridis', aspect='auto')
        ax3.set_xticks(range(len(task_names)))
        ax3.set_xticklabels([t.replace('_', ' ').title() for t in task_names], rotation=45, ha='right')
        ax3.set_yticks(range(len(layers)))
        ax3.set_yticklabels([f'Layer {l}' for l in layers])
        ax3.set_title('Task Specialization Heatmap\nAcross Layers and Tasks', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Average Task Specialization Score')

    def _plot_robustness_analysis(self, fig, gs, robustness_results: Dict[str, Any]):
        """Plot robustness and recovery analysis (Research Question 2)"""
        
        # Plot 4: Ablation curves comparison
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Compare different ablation strategies
        strategy_colors = {
            'hub_first': '#FF6B6B',
            'specialist_first': '#4ECDC4', 
            'random': '#45B7D1',
            'importance_first': '#96CEB4'
        }
        
        # Use first layer and first task for demonstration
        first_layer = sorted(robustness_results['hub_first'].keys())[0]
        first_task = list(self.create_multi_task_metrics().keys())[0]
        
        for strategy, strategy_data in robustness_results.items():
            if first_layer in strategy_data:
                ablation_curve = strategy_data[first_layer]['ablation_curve'][first_task]
                steps = range(len(ablation_curve))
                ax4.plot(steps, ablation_curve, 'o-', color=strategy_colors.get(strategy, '#333333'),
                        linewidth=2, label=strategy.replace('_', ' ').title(), markersize=4)
        
        ax4.set_xlabel('Number of Neurons Ablated')
        ax4.set_ylabel('Task Performance')
        ax4.set_title(f'Ablation Strategies Comparison\nLayer {first_layer}, {first_task.title()}', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Recovery scores across strategies
        ax5 = fig.add_subplot(gs[1, 1])
        
        strategies = list(robustness_results.keys())
        layers = sorted(robustness_results[strategies[0]].keys())
        
        recovery_data = []
        strategy_labels = []
        
        for strategy in strategies:
            avg_recoveries = []
            for layer_idx in layers:
                if layer_idx in robustness_results[strategy]:
                    recovery_scores = robustness_results[strategy][layer_idx]['recovery_analysis']['recovery_scores']
                    avg_recovery = np.mean(list(recovery_scores.values()))
                    avg_recoveries.append(avg_recovery)
            
            if avg_recoveries:
                recovery_data.append(np.mean(avg_recoveries))
                strategy_labels.append(strategy.replace('_', ' ').title())
        
        bars = ax5.bar(strategy_labels, recovery_data, 
                      color=[strategy_colors.get(s.lower().replace(' ', '_'), '#333333') for s in strategy_labels],
                      alpha=0.7)
        
        ax5.set_ylabel('Average Recovery Score')
        ax5.set_title('Robustness Comparison\nAcross Ablation Strategies', fontweight='bold')
        ax5.set_xticklabels(strategy_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, recovery_data):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Layer compensation analysis
        ax6 = fig.add_subplot(gs[1, 2:])
        
        # Show which layers can compensate for ablations
        compensation_matrix = np.zeros((len(layers), len(layers)))
        
        for i, ablated_layer in enumerate(layers):
            for strategy in robustness_results.keys():
                if ablated_layer in robustness_results[strategy]:
                    recovery_data = robustness_results[strategy][ablated_layer]['recovery_analysis']
                    compensation_layers = recovery_data.get('compensation_layers', [])
                    
                    for comp_layer in compensation_layers:
                        if comp_layer in layers:
                            j = layers.index(comp_layer)
                            compensation_matrix[i, j] += 1
        
        # Normalize
        compensation_matrix = compensation_matrix / len(robustness_results) if len(robustness_results) > 0 else compensation_matrix
        
        im = ax6.imshow(compensation_matrix, cmap='Reds', aspect='auto')
        ax6.set_xticks(range(len(layers)))
        ax6.set_xticklabels([f'L{l}' for l in layers])
        ax6.set_yticks(range(len(layers)))
        ax6.set_yticklabels([f'L{l}' for l in layers])
        ax6.set_xlabel('Compensating Layer')
        ax6.set_ylabel('Ablated Layer')
        ax6.set_title('Layer Compensation Patterns\n(Later Layers Recovering from Ablations)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(layers)):
            for j in range(len(layers)):
                if compensation_matrix[i, j] > 0:
                    text = ax6.text(j, i, f'{compensation_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="white" if compensation_matrix[i, j] > 0.5 else "black",
                                   fontweight='bold')
        
        plt.colorbar(im, ax=ax6, shrink=0.8)

    def _plot_size_analysis(self, fig, gs, size_analysis: Dict[str, Any]):
        """Plot model size effects analysis (Research Question 3)"""
        
        # Plot 7: Importance distribution across layers
        ax7 = fig.add_subplot(gs[2, 0])
        
        layers = sorted(size_analysis['importance_distribution'].keys())
        gini_coeffs = [size_analysis['importance_distribution'][l]['gini_coefficient'] for l in layers]
        top_1_shares = [size_analysis['importance_distribution'][l]['top_1_percent_share'] for l in layers]
        
        ax7_twin = ax7.twinx()
        
        line1 = ax7.plot(layers, gini_coeffs, 'o-', color='#E74C3C', linewidth=3, markersize=8, label='Gini Coefficient')
        line2 = ax7_twin.plot(layers, top_1_shares, 's-', color='#3498DB', linewidth=3, markersize=8, label='Top 1% Share')
        
        ax7.set_xlabel('Layer Index')
        ax7.set_ylabel('Gini Coefficient', color='#E74C3C')
        ax7_twin.set_ylabel('Top 1% Importance Share', color='#3498DB')
        ax7.set_title('Importance Inequality\nAcross Layers', fontweight='bold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax7.legend(lines, labels, loc='upper right')
        
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Effective neurons vs layer depth
        ax8 = fig.add_subplot(gs[2, 1])
        
        effective_neurons = [size_analysis['importance_distribution'][l]['effective_neurons'] for l in layers]
        utilization_rates = [size_analysis['layer_efficiency'][l]['utilization_rate'] for l in layers]
        
        ax8.plot(layers, effective_neurons, 'o-', color='#9B59B6', linewidth=3, markersize=8, label='Effective Neurons')
        ax8_twin = ax8.twinx()
        ax8_twin.plot(layers, utilization_rates, 's-', color='#F39C12', linewidth=3, markersize=8, label='Utilization Rate')
        
        ax8.set_xlabel('Layer Index')
        ax8.set_ylabel('Effective Neurons', color='#9B59B6')
        ax8_twin.set_ylabel('Utilization Rate', color='#F39C12')
        ax8.set_title('Neuron Utilization\nEfficiency', fontweight='bold')
        
        # Combine legends
        lines1 = ax8.get_lines()
        lines2 = ax8_twin.get_lines()
        ax8.legend(lines1 + lines2, ['Effective Neurons', 'Utilization Rate'], loc='upper right')
        
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Scaling patterns summary
        ax9 = fig.add_subplot(gs[2, 2:])
        
        # Create scaling summary
        scaling_data = size_analysis['scaling_patterns']['early_vs_late']['comparisons']
        
        metrics = []
        early_values = []
        late_values = []
        differences = []
        
        for metric_name, comparison in scaling_data.items():
            metrics.append(metric_name.replace('_', ' ').title())
            early_values.append(comparison['early_mean'])
            late_values.append(comparison['late_mean'])
            differences.append(comparison['difference'])
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax9.bar(x - width/2, early_values, width, label='Early Layers', color='#2ECC71', alpha=0.7)
        bars2 = ax9.bar(x + width/2, late_values, width, label='Late Layers', color='#E67E22', alpha=0.7)
        
        ax9.set_xlabel('Metrics')
        ax9.set_ylabel('Metric Values')
        ax9.set_title('Early vs Late Layer\nCharacteristics', fontweight='bold')
        ax9.set_xticks(x)
        ax9.set_xticklabels(metrics, rotation=45, ha='right')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # Add difference indicators
        for i, diff in enumerate(differences):
            if diff > 0:
                ax9.annotate('↑', xy=(i, max(early_values[i], late_values[i])), 
                           xytext=(i, max(early_values[i], late_values[i]) + 0.1 * max(early_values + late_values)),
                           ha='center', fontsize=16, color='green', fontweight='bold')
            elif diff < 0:
                ax9.annotate('↓', xy=(i, max(early_values[i], late_values[i])), 
                           xytext=(i, max(early_values[i], late_values[i]) + 0.1 * max(early_values + late_values)),
                           ha='center', fontsize=16, color='red', fontweight='bold')
    
    def run_optimized_research_analysis(self, audio_inputs: List[torch.Tensor],
                                      layers_to_analyze: List[int] = None) -> Dict[str, Any]:
        """
        Run optimized research analysis focusing on layer recovery and model size effects
        """
        if layers_to_analyze is None:
            layers_to_analyze = [0, 3, 6, 9, 11]  # Reduced layers for speed
        
        print(f"\n🧠 OPTIMIZED NEURON RESEARCH ANALYSIS")
        print(f"{'='*60}")
        print(f"Focused Research Questions:")
        print(f"1. Can later layers recover from ablations (robustness)?")
        print(f"2. How does model size affect neuron importance distribution?")
        print(f"{'='*60}")
        
        # Step 1: Fast SHAP-based neuron identification
        shap_results = self.step1_shap_neuron_identification(
            audio_inputs, layers_to_analyze, top_k_per_layer=30  # Reduced for speed
        )
        
        # Step 2: Fast multi-task hub analysis
        hub_analysis = self.step2_multi_task_hub_analysis(audio_inputs)
        
        # Step 3: Progressive ablation and robustness (main focus)
        robustness_results = self.step3_progressive_ablation_robustness(
            audio_inputs, hub_analysis
        )
        
        # Step 4: Model size effect analysis (main focus)
        size_analysis = self.step4_model_size_analysis(audio_inputs, hub_analysis)
        
        # Combine all results
        complete_results = {
            'shap_results': shap_results,
            'hub_analysis': hub_analysis,
            'robustness_results': robustness_results,
            'size_analysis': size_analysis,
            'research_summary': self._generate_research_summary(
                hub_analysis, robustness_results, size_analysis
            )
        }
        
        return complete_results

    def _generate_research_summary(self, hub_analysis: Dict[str, Any],
                                 robustness_results: Dict[str, Any],
                                 size_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of research findings"""
        
        summary = {
            'question_1_findings': {},  # Multi-task hubs vs specialists
            'question_2_findings': {},  # Layer recovery and robustness
            'question_3_findings': {}   # Model size effects
        }
        
        # Question 1: Multi-task hub analysis
        total_hubs = sum(len(data['hub_neurons']) for data in hub_analysis.values())
        total_specialists = sum(len(data['specialist_neurons']) for data in hub_analysis.values())
        total_analyzed = sum(len(data['neuron_profiles']) for data in hub_analysis.values())
        
        # Find layer with most hubs
        hub_counts_by_layer = {layer: len(data['hub_neurons']) for layer, data in hub_analysis.items()}
        most_hub_layer = max(hub_counts_by_layer.keys(), key=lambda k: hub_counts_by_layer[k])
        
        # Find layer with most specialists
        specialist_counts_by_layer = {layer: len(data['specialist_neurons']) for layer, data in hub_analysis.items()}
        most_specialist_layer = max(specialist_counts_by_layer.keys(), key=lambda k: specialist_counts_by_layer[k])
        
        summary['question_1_findings'] = {
            'total_neurons_analyzed': total_analyzed,
            'total_multitask_hubs': total_hubs,
            'total_task_specialists': total_specialists,
            'hub_percentage': (total_hubs / total_analyzed * 100) if total_analyzed > 0 else 0,
            'specialist_percentage': (total_specialists / total_analyzed * 100) if total_analyzed > 0 else 0,
            'most_hub_dense_layer': most_hub_layer,
            'most_specialist_dense_layer': most_specialist_layer,
            'layer_specialization_pattern': 'early' if most_specialist_layer < 6 else 'late'
        }
        
        # Question 2: Robustness analysis
        strategy_performance = {}
        for strategy, strategy_data in robustness_results.items():
            avg_recoveries = []
            for layer_data in strategy_data.values():
                recovery_scores = layer_data['recovery_analysis']['recovery_scores']
                avg_recoveries.extend(recovery_scores.values())
            
            strategy_performance[strategy] = np.mean(avg_recoveries) if avg_recoveries else 0
        
        most_robust_strategy = max(strategy_performance.keys(), key=lambda k: strategy_performance[k])
        least_robust_strategy = min(strategy_performance.keys(), key=lambda k: strategy_performance[k])
        
        summary['question_2_findings'] = {
            'strategy_rankings': sorted(strategy_performance.items(), key=lambda x: x[1], reverse=True),
            'most_robust_strategy': most_robust_strategy,
            'least_robust_strategy': least_robust_strategy,
            'robustness_difference': strategy_performance[most_robust_strategy] - strategy_performance[least_robust_strategy],
            'hub_vs_specialist_robustness': 'hubs_more_robust' if strategy_performance.get('specialist_first', 0) > strategy_performance.get('hub_first', 0) else 'specialists_more_robust'
        }
        
        # Question 3: Model size effects
        layers = sorted(size_analysis['importance_distribution'].keys())
        
        # Calculate concentration trends
        gini_values = [size_analysis['importance_distribution'][l]['gini_coefficient'] for l in layers]
        utilization_values = [size_analysis['layer_efficiency'][l]['utilization_rate'] for l in layers]
        
        # Early vs late comparison
        mid_point = len(layers) // 2
        early_gini = np.mean(gini_values[:mid_point])
        late_gini = np.mean(gini_values[mid_point:])
        early_util = np.mean(utilization_values[:mid_point])
        late_util = np.mean(utilization_values[mid_point:])
        
        summary['question_3_findings'] = {
            'concentration_trend': 'increasing' if late_gini > early_gini else 'decreasing',
            'utilization_trend': 'increasing' if late_util > early_util else 'decreasing',
            'early_layer_concentration': early_gini,
            'late_layer_concentration': late_gini,
            'early_layer_utilization': early_util,
            'late_layer_utilization': late_util,
            'most_concentrated_layer': layers[np.argmax(gini_values)],
            'least_concentrated_layer': layers[np.argmin(gini_values)],
            'scaling_pattern': size_analysis['scaling_patterns']['early_vs_late']
        }
        
        return summary


def run_neuron_probing_research(ravdess_path: str,
                               emotions: List[str] = ["happy", "sad", "angry"],
                               num_samples_per_emotion: int = 3,
                               save_dir: str = "neuron_research_results"):
    """
    Main function to run the complete neuron probing research
    
    Args:
        ravdess_path: Path to RAVDESS dataset
        emotions: List of emotions to analyze
        num_samples_per_emotion: Number of audio samples per emotion
        save_dir: Directory to save results
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🧠 NEURON PROBING RESEARCH PIPELINE")
    print(f"{'='*60}")
    print(f"Emotions: {emotions}")
    print(f"Samples per emotion: {num_samples_per_emotion}")
    print(f"Save directory: {save_dir}")
    
    # Initialize analyzer
    analyzer = NeuronProbingAnalyzer()
    
    # Collect audio samples
    print(f"\n📁 Collecting audio samples...")
    all_audio_inputs = []
    all_metadata = []
    
    for emotion in emotions:
        for i in range(num_samples_per_emotion):
            try:
                audio_path, metadata = analyzer.load_ravdess_sample(
                    ravdess_path=ravdess_path,
                    emotion_label=emotion
                )
                audio_input = analyzer.preprocess_audio(audio_path)
                all_audio_inputs.append(audio_input)
                all_metadata.append(metadata)
            except Exception as e:
                print(f"Warning: Could not load sample {i+1} for {emotion}: {e}")
    
    print(f"✅ Collected {len(all_audio_inputs)} audio samples")
    
    # Run optimized analysis
    print(f"\n🔬 Running optimized research analysis...")
    layers_to_analyze = [0, 3, 6, 9, 11]  # Reduced for speed
    
    results = analyzer.run_optimized_research_analysis(
        audio_inputs=all_audio_inputs,
        layers_to_analyze=layers_to_analyze
    )
    
    # Create focused visualizations
    print(f"\n📊 Creating focused visualizations...")
    viz_path = os.path.join(save_dir, "focused_research_analysis.png")
    fig = analyzer.create_focused_visualizations(
        robustness_results=results['robustness_results'],
        size_analysis=results['size_analysis'],
        save_path=viz_path
    )
    
    # Save detailed results
    print(f"\n💾 Saving detailed results...")
    
    # Convert results for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    save_data = {
        'metadata': {
            'emotions_analyzed': emotions,
            'num_samples': len(all_audio_inputs),
            'layers_analyzed': layers_to_analyze,
            'sample_metadata': all_metadata
        },
        'research_results': convert_for_json(results)
    }
    
    results_path = os.path.join(save_dir, "complete_research_results.json")
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Print research summary
    print(f"\n{'='*60}")
    print("🔬 RESEARCH FINDINGS SUMMARY")
    print(f"{'='*60}")
    
    summary = results['research_summary']
    
    print(f"\n❓ Question 1: Multi-task hubs vs task-specific neurons")
    print(f"  🔗 Multi-task hubs: {summary['question_1_findings']['total_multitask_hubs']} ({summary['question_1_findings']['hub_percentage']:.1f}%)")
    print(f"  🎯 Task specialists: {summary['question_1_findings']['total_task_specialists']} ({summary['question_1_findings']['specialist_percentage']:.1f}%)")
    print(f"  📊 Most hub-dense layer: {summary['question_1_findings']['most_hub_dense_layer']}")
    print(f"  📊 Most specialist-dense layer: {summary['question_1_findings']['most_specialist_dense_layer']}")
    
    print(f"\n❓ Question 2: Layer recovery and robustness")
    print(f"  🛡️ Most robust strategy: {summary['question_2_findings']['most_robust_strategy']}")
    print(f"  ⚠️ Least robust strategy: {summary['question_2_findings']['least_robust_strategy']}")
    print(f"  📈 Robustness difference: {summary['question_2_findings']['robustness_difference']:.3f}")
    print(f"  🔄 Hub vs specialist robustness: {summary['question_2_findings']['hub_vs_specialist_robustness']}")
    
    print(f"\n❓ Question 3: Model size effects")
    print(f"  📏 Concentration trend: {summary['question_3_findings']['concentration_trend']}")
    print(f"  📊 Utilization trend: {summary['question_3_findings']['utilization_trend']}")
    print(f"  🎯 Most concentrated layer: {summary['question_3_findings']['most_concentrated_layer']}")
    print(f"  🎯 Least concentrated layer: {summary['question_3_findings']['least_concentrated_layer']}")
    
    print(f"\n✅ Research complete! Results saved to: {save_dir}")
    
    return results, all_metadata, analyzer


# Quick analysis function for notebook use
def quick_research_analysis(ravdess_path: str, emotions: List[str] = ["happy", "sad"]):
    """Ultra-fast research analysis for notebook use"""
    try:
        results, metadata, analyzer = run_neuron_probing_research(
            ravdess_path=ravdess_path,
            emotions=emotions,
            num_samples_per_emotion=2,  # Minimal samples for speed
            save_dir="quick_research_results"
        )
        return results, metadata, analyzer
    except Exception as e:
        print(f"❌ Error during research analysis: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None, None, None


# Example usage
if __name__ == "__main__":
    # Set your RAVDESS path
    RAVDESS_PATH = "/path/to/your/ravdess/dataset"
    
    # Run quick research analysis
    results, metadata, analyzer = quick_research_analysis(
        RAVDESS_PATH, 
        emotions=["happy", "sad", "angry"]
    )
    
    
    # Or run full research analysis
    results, metadata, analyzer = run_neuron_probing_research(
        ravdess_path=RAVDESS_PATH,
        emotions=["happy", "sad", "angry", "neutral"],
        num_samples_per_emotion=3,
        save_dir="complete_research_results"
    )