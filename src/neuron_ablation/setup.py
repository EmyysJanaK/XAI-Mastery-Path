"""
Setup script for the neuron_ablation package.
"""

from setuptools import setup, find_packages

setup(
    name="neuron_ablation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.5",
        "matplotlib>=3.3.4",
        "seaborn>=0.11.1",
        "tqdm>=4.62.3",
        "librosa>=0.8.1",
        "transformers>=4.12.0",
        "torchaudio>=0.9.0",
        "scikit-learn>=0.24.2",
    ],
    author="XAI Research Team",
    author_email="example@example.com",
    description="A framework for neuron ablation analysis in speech models",
    keywords="explainable-ai, ablation-studies, speech-processing, transformer-models",
    python_requires=">=3.7",
)