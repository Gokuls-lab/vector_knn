"""Base encoder interface for all embedding models."""

from abc import ABC, abstractmethod
from typing import Union, List, Any
import numpy as np


class BaseEncoder(ABC):
    """Abstract base class for all encoders.
    
    All encoder implementations must inherit from this class and implement
    the encode() method.
    """
    
    def __init__(self, model_name: str = None, device: str = 'cpu', **kwargs):
        """Initialize encoder.
        
        Args:
            model_name: Name/path of the model
            device: Device for computation
            **kwargs: Additional encoder-specific parameters
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.dimension = None
        self.is_loaded = False
        
    @abstractmethod
    def load(self) -> None:
        """Load the encoder model."""
        pass
    
    @abstractmethod
    def encode(
        self, 
        inputs: Union[List[Any], Any],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode inputs to embeddings.
        
        Args:
            inputs: Input data (text, images, audio, etc.)
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings
            **kwargs: Additional encoding parameters
            
        Returns:
            Embeddings as numpy array (n_samples, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"