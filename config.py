"""Configuration management for Vector KNN Classifier."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os


@dataclass
class Config:
    """Global configuration for VectorKNN.
    
    Attributes:
        model_dir: Directory to save/load models
        k_neighbors: Default number of neighbors
        random_state: Random seed
        test_size: Test split proportion
        val_size: Validation split proportion
        cache_dir: Cache directory for models
        device: Device for computation ('cuda', 'cpu', 'mps')
        batch_size: Batch size for encoding
        show_progress: Show progress bars
    """
    
    # Paths
    model_dir: str = "saved_models"
    cache_dir: str = "~/.cache/vector_knn"
    
    # Model parameters
    embedding_model: str = "all-MiniLM-L6-v2"
    k_neighbors: int = 5
    random_state: int = 42
    
    # Data splitting
    test_size: float = 0.2
    val_size: float = 0.2
    
    # Computation
    device: Optional[str] = None  # Auto-detect if None
    batch_size: int = 32
    show_progress: bool = True
    
    # Additional settings
    index_type: str = "Flat"  # 'Flat', 'IVF', 'HNSW'
    nlist: int = 100  # For IVF
    normalize_embeddings: bool = True
    distance_metric: str = "cosine"  # 'cosine', 'euclidean', 'dot'
    
    # Encoder-specific settings
    encoder_config: Dict[str, Any] = field(default_factory=dict)
    
    # Index-specific settings
    index_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Expand paths
        self.cache_dir = os.path.expanduser(self.cache_dir)
        
        # Auto-detect device
        if self.device is None:
            self.device = self._auto_detect_device()
    
    @staticmethod
    def _auto_detect_device() -> str:
        """Auto-detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
        except ImportError:
            pass
        return 'cpu'
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < self.val_size < 1:
            raise ValueError("val_size must be between 0 and 1")
        if self.distance_metric not in ['cosine', 'euclidean', 'dot', 'l2']:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")


# Default presets for common use cases
PRESET_CONFIGS = {
    'fast': Config(
        embedding_model='all-MiniLM-L6-v2',
        k_neighbors=5,
        batch_size=64,
    ),
    'balanced': Config(
        embedding_model='all-MiniLM-L12-v2',
        k_neighbors=7,
        batch_size=32,
    ),
    'accurate': Config(
        embedding_model='all-mpnet-base-v2',
        k_neighbors=10,
        batch_size=16,
    ),
    'multilingual': Config(
        embedding_model='paraphrase-multilingual-mpnet-base-v2',
        k_neighbors=7,
        batch_size=32,
    ),
}


def get_preset_config(preset: str) -> Config:
    """Get a preset configuration.
    
    Args:
        preset: Preset name ('fast', 'balanced', 'accurate', 'multilingual')
        
    Returns:
        Config object
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[preset]