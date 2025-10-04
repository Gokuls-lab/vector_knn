"""
Vector Space K-Nearest Neighbors Classifier

A highly customizable KNN classifier using vector embeddings and similarity search.
Supports multiple modalities (text, image, audio, video) and backends (FAISS, Pinecone, Milvus, etc.)
"""

from .core.base_classifier import VectorKNNClassifier
from .core.base_encoder import BaseEncoder
from .core.base_index import BaseIndex

# Import optimizers
# from .optimizers import auto_optimize_k, auto_optimize_hyperparameters

# Pre-configured classifiers
from .classifiers.presets import (
    FAISSKNNClassifier,
    FAISSKNNClusterer,
    PineconeKNNClassifier,
    ConvNeXtClassifier,  # Add ConvNeXt classifier
    MilvusKNNClassifier,
    AnnoyKNNClassifier,
    ChromaKNNClassifier,
    TextFAISSClassifier,
    ImageFAISSClassifier,
    AudioFAISSClassifier,
    VideoFAISSClassifier,
    MultimodalFAISSClassifier,
    ImageResNetClassifier,
)

# Encoders
from .encoders import (
    SentenceTransformerEncoder,
    OpenAIEncoder,
    HuggingFaceEncoder,
    CLIPImageEncoder,
    ResNetEncoder,
    ViTEncoder,
    Wav2VecEncoder,
    WhisperEncoder,
    VideoMAEEncoder,
    CLIPMultimodalEncoder,
)

# Indexes
from .indexes import (
    FAISSIndex,
    PineconeIndex,
    MilvusIndex,
    AnnoyIndex,
    ChromaDBIndex,
)

# Optimizers
from .optimizers import KOptimizer, auto_optimize_k, auto_optimize_hyperparameters

# Config
from .config import Config, get_preset_config

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    # Core
    "VectorKNNClassifier",
    "BaseEncoder",
    "BaseIndex",
    "Config",
    "get_preset_config",
    
    # Optimization
    "auto_optimize_k",
    "auto_optimize_hyperparameters",
    
    # Presets
    "FAISSKNNClassifier",
    "FAISSKNNClusterer",
    "PineconeKNNClassifier",
    "MilvusKNNClassifier",
    "AnnoyKNNClassifier",
    "ChromaKNNClassifier",
    "TextFAISSClassifier",
    "ImageFAISSClassifier",
    "AudioFAISSClassifier",
    "VideoFAISSClassifier",
    "MultimodalFAISSClassifier",
    "ImageResNetClassifier",
    
    # Encoders
    "SentenceTransformerEncoder",
    "OpenAIEncoder",
    "HuggingFaceEncoder",
    "CLIPImageEncoder",
    "ResNetEncoder",
    "ViTEncoder",
    "Wav2VecEncoder",
    "WhisperEncoder",
    "VideoMAEEncoder",
    "CLIPMultimodalEncoder",
    
    # Indexes
    "FAISSIndex",
    "PineconeIndex",
    "MilvusIndex",
    "AnnoyIndex",
    "ChromaDBIndex",
    
    # Optimizers
    "KOptimizer",
    "auto_optimize_k",
]