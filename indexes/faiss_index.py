"""FAISS index implementation."""

from typing import Tuple
import numpy as np
import pickle
import os
from ..core.base_index import BaseIndex


class FAISSIndex(BaseIndex):
    """FAISS-based vector search index.
    
    Fast, efficient, and supports multiple distance metrics.
    Best for: Local deployment, fast similarity search.
    
    Example:
        >>> index = FAISSIndex(dimension=384, metric='cosine')
        >>> index.build(embeddings)
        >>> distances, indices = index.search(query_embeddings, k=5)
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = 'cosine',
        index_type: str = 'Flat',  # 'Flat', 'IVF', 'HNSW'
        nlist: int = 100,  # For IVF
        **kwargs
    ):
        super().__init__(dimension, metric, **kwargs)
        self.index_type = index_type
        self.nlist = nlist
        
    def build(self, embeddings: np.ndarray) -> None:
        """Build FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError("Please install: pip install faiss-cpu (or faiss-gpu)")
        
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
            metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric in ['euclidean', 'l2']:
            metric = faiss.METRIC_L2
        else:
            raise ValueError(f"Unsupported metric for FAISS: {self.metric}")
        
        # Build index based on type
        print(f"Building FAISS index of type '{self.index_type}' with metric '{self.metric}'...")
        if self.index_type == 'Flat':
            if metric == faiss.METRIC_INNER_PRODUCT:
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
                
        elif self.index_type == 'IVF':
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric)
            self.index.train(embeddings)
            
        elif self.index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, metric)
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add vectors
        self.index.add(embeddings)
        self.n_vectors = self.index.ntotal
        self.is_built = True
        
        print(f"✓ FAISS index built: {self.n_vectors} vectors")
    
    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to index."""
        import faiss
        
        if not self.is_built:
            raise ValueError("Index not built. Call build() first.")
        
        embeddings = embeddings.astype('float32')
        
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.n_vectors = self.index.ntotal
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        import faiss
        
        if not self.is_built:
            raise ValueError("Index not built. Call build() first.")
        
        query_embeddings = query_embeddings.astype('float32')
        
        if self.metric == 'cosine':
            faiss.normalize_L2(query_embeddings)
        
        distances, indices = self.index.search(query_embeddings, k)
        
        return distances, indices
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        import faiss
        
        if not self.is_built:
            raise ValueError("Index not built. Cannot save.")
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save metadata
        metadata = self.get_config()
        metadata['index_type'] = self.index_type
        metadata['nlist'] = self.nlist
        
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        import faiss
        
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Load metadata
        with open(f"{path}.meta", 'rb') as f:
            metadata = pickle.load(f)
        
        self.dimension = metadata['dimension']
        self.metric = metadata['metric']
        self.n_vectors = metadata['n_vectors']
        self.index_type = metadata['index_type']
        self.nlist = metadata.get('nlist', 100)
        self.is_built = True