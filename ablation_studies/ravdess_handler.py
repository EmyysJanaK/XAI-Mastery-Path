#!/usr/bin/env python3

import os
import torch
import torchaudio
import pandas as pd
from typing import List, Tuple, Optional, Dict
import random

class RAVDESSDataHandler:
    """
    Handler for RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset
    """
    
    def __init__(self, data_path: str):
        """
        Initialize RAVDESS data handler
        
        Args:
            data_path: Path to RAVDESS dataset directory
        """
        self.data_path = data_path
        self.emotion_map = {
            1: "neutral", 2: "calm", 3: "happy", 4: "sad",
            5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
        }
        self.intensity_map = {1: "normal", 2: "strong"}
        self.statement_map = {1: "Kids are talking by the door", 2: "Dogs are sitting by the door"}
        
        # Scan for available files
        self.audio_files = self._scan_audio_files()
        print(f"Found {len(self.audio_files)} RAVDESS audio files")
    
    def _scan_audio_files(self) -> List[Dict]:
        """
        Scan directory for RAVDESS audio files and extract metadata
        
        Returns:
            List of file information dictionaries
        """
        audio_files = []
        
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav') and file.startswith('03-'):  # Speech files start with 03
                    file_path = os.path.join(root, file)
                    metadata = self._parse_filename(file)
                    if metadata:
                        metadata['file_path'] = file_path
                        audio_files.append(metadata)
        
        return audio_files
    
    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """
        Parse RAVDESS filename to extract metadata
        
        RAVDESS filename format:
        Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
        
        Args:
            filename: RAVDESS filename
            
        Returns:
            Dictionary with metadata or None if parsing fails
        """
        try:
            parts = filename.replace('.wav', '').split('-')
            if len(parts) != 7:
                return None
            
            modality = int(parts[0])  # 03 = Audio-only
            vocal_channel = int(parts[1])  # 01 = speech, 02 = song
            emotion = int(parts[2])  # 01-08
            intensity = int(parts[3])  # 01 = normal, 02 = strong
            statement = int(parts[4])  # 01-02
            repetition = int(parts[5])  # 01-02
            actor = int(parts[6])  # 01-24 (odd = male, even = female)
            
            return {
                'filename': filename,
                'modality': modality,
                'vocal_channel': vocal_channel,
                'emotion_id': emotion,
                'emotion': self.emotion_map.get(emotion, 'unknown'),
                'intensity_id': intensity,
                'intensity': self.intensity_map.get(intensity, 'unknown'),
                'statement_id': statement,
                'statement': self.statement_map.get(statement, 'unknown'),
                'repetition': repetition,
                'actor': actor,
                'gender': 'male' if actor % 2 == 1 else 'female'
            }
        except (ValueError, IndexError):
            return None
    
    def get_files_by_emotion(self, emotion: str) -> List[Dict]:
        """
        Get all files for a specific emotion
        
        Args:
            emotion: Emotion name (e.g., 'happy', 'sad', 'angry')
            
        Returns:
            List of file information for the specified emotion
        """
        return [f for f in self.audio_files if f['emotion'] == emotion]
    
    def get_files_by_actor(self, actor_id: int) -> List[Dict]:
        """
        Get all files for a specific actor
        
        Args:
            actor_id: Actor ID (1-24)
            
        Returns:
            List of file information for the specified actor
        """
        return [f for f in self.audio_files if f['actor'] == actor_id]
    
    def get_random_sample(self, 
                         emotion: Optional[str] = None,
                         gender: Optional[str] = None,
                         intensity: Optional[str] = None) -> Dict:
        """
        Get a random audio sample with optional filtering
        
        Args:
            emotion: Filter by emotion (optional)
            gender: Filter by gender ('male'/'female', optional)
            intensity: Filter by intensity ('normal'/'strong', optional)
            
        Returns:
            Random file information dictionary
        """
        filtered_files = self.audio_files.copy()
        
        if emotion:
            filtered_files = [f for f in filtered_files if f['emotion'] == emotion]
        if gender:
            filtered_files = [f for f in filtered_files if f['gender'] == gender]
        if intensity:
            filtered_files = [f for f in filtered_files if f['intensity'] == intensity]
        
        if not filtered_files:
            raise ValueError("No files match the specified criteria")
        
        return random.choice(filtered_files)
    
    def load_audio(self, file_info: Dict, target_sr: int = 16000) -> Tuple[torch.Tensor, Dict]:
        """
        Load audio file and return waveform with metadata
        
        Args:
            file_info: File information dictionary
            target_sr: Target sampling rate
            
        Returns:
            Tuple of (waveform tensor, metadata)
        """
        waveform, sample_rate = torchaudio.load(file_info['file_path'])
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform, file_info
    
    def get_emotion_distribution(self) -> Dict[str, int]:
        """
        Get distribution of emotions in the dataset
        
        Returns:
            Dictionary with emotion counts
        """
        emotion_counts = {}
        for file_info in self.audio_files:
            emotion = file_info['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        return emotion_counts
    
    def print_dataset_summary(self):
        """
        Print summary statistics of the dataset
        """
        print("RAVDESS Dataset Summary")
        print("=" * 30)
        print(f"Total audio files: {len(self.audio_files)}")
        
        # Emotion distribution
        emotion_dist = self.get_emotion_distribution()
        print("\nEmotion distribution:")
        for emotion, count in emotion_dist.items():
            print(f"  {emotion}: {count}")
        
        # Gender distribution
        gender_dist = {'male': 0, 'female': 0}
        for file_info in self.audio_files:
            gender_dist[file_info['gender']] += 1
        print(f"\nGender distribution:")
        print(f"  Male: {gender_dist['male']}")
        print(f"  Female: {gender_dist['female']}")
        
        # Intensity distribution
        intensity_dist = {'normal': 0, 'strong': 0}
        for file_info in self.audio_files:
            intensity_dist[file_info['intensity']] += 1
        print(f"\nIntensity distribution:")
        print(f"  Normal: {intensity_dist['normal']}")
        print(f"  Strong: {intensity_dist['strong']}")


# Example usage
if __name__ == "__main__":
    # Initialize handler (replace with your RAVDESS path)
    ravdess_path = "/path/to/ravdess/audio_speech_actors_01-24/"
    
    try:
        handler = RAVDESSDataHandler(ravdess_path)
        handler.print_dataset_summary()
        
        # Get a random happy sample
        happy_sample = handler.get_random_sample(emotion="happy")
        print(f"\nRandom happy sample: {happy_sample['filename']}")
        
        # Load the audio
        waveform, metadata = handler.load_audio(happy_sample)
        print(f"Audio shape: {waveform.shape}")
        print(f"Metadata: {metadata}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set the correct path to your RAVDESS dataset")