"""Base index interface for all vector search backends."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np


class BaseIndex(ABC):
    """Abstract base class for all vector search indexes.
    
    All index implementations must inherit from this class.
    """
    
    def __init__(self, dimension: int, metric: str = 'cosine', **kwargs):
        """Initialize index.
        
        Args:
            dimension: Embedding dimension
            metric: Distance metric ('cosine', 'euclidean', 'dot')
            **kwargs: Additional index-specific parameters
        """
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.is_built = False
        self.n_vectors = 0
        
    @abstractmethod
    def build(self, embeddings: np.ndarray) -> None:
        """Build index from embeddings.
        
        Args:
            embeddings: Training embeddings (n_samples, dimension)
        """
        pass
    
    @abstractmethod
    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to existing index.
        
        Args:
            embeddings: Embeddings to add (n_samples, dimension)
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query_embeddings: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.
        
        Args:
            query_embeddings: Query embeddings (n_queries, dimension)
            k: Number of neighbors
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save index to disk.
        
        Args:
            path: Save path
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk.
        
        Args:
            path: Load path
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get index configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'dimension': self.dimension,
            'metric': self.metric,
            'n_vectors': self.n_vectors,
            'index_type': self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimension={self.dimension}, metric='{self.metric}')"