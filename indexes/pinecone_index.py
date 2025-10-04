"""Pinecone index implementation."""

from typing import Tuple, Optional
import numpy as np
import pickle
from ..core.base_index import BaseIndex


class PineconeIndex(BaseIndex):
    """Pinecone vector database index.
    
    Cloud-based, managed vector database.
    Best for: Production deployments, scalability, managed infrastructure.
    
    Example:
        >>> index = PineconeIndex(
        ...     dimension=384,
        ...     metric='cosine',
        ...     api_key='your-api-key',
        ...     environment='us-west1-gcp',
        ...     index_name='my-index'
        ... )
        >>> index.build(embeddings)
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = 'cosine',
        api_key: Optional[str] = None,
        environment: str = 'us-west1-gcp',
        index_name: str = 'vector-knn-index',
        **kwargs
    ):
        super().__init__(dimension, metric, **kwargs)
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pinecone = None
        
    def _init_pinecone(self):
        """Initialize Pinecone client."""
        try:
            import pinecone
        except ImportError:
            raise ImportError("Please install: pip install pinecone-client")
        
        pinecone.init(api_key=self.api_key, environment=self.environment)
        self.pinecone = pinecone
        
    def build(self, embeddings: np.ndarray) -> None:
        """Build Pinecone index."""
        self._init_pinecone()
        
        # Create index if doesn't exist
        if self.index_name not in self.pinecone.list_indexes():
            self.pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric
            )
        
        # Connect to index
        self.index = self.pinecone.Index(self.index_name)
        
        # Upsert vectors
        vectors = [(str(i), emb.tolist()) for i, emb in enumerate(embeddings)]
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        self.n_vectors = len(embeddings)
        self.is_built = True
        
        print(f"✓ Pinecone index built: {self.n_vectors} vectors")
    
    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to index."""
        if not self.is_built:
            raise ValueError("Index not built. Call build() first.")
        
        start_id = self.n_vectors
        vectors = [(str(start_id + i), emb.tolist()) for i, emb in enumerate(embeddings)]
        
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        self.n_vectors += len(embeddings)
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        if not self.is_built:
            raise ValueError("Index not built. Call build() first.")
        
        all_distances = []
        all_indices = []
        
        for query in query_embeddings:
            results = self.index.query(
                vector=query.tolist(),
                top_k=k,
                include_values=False
            )
            
            distances = [match['score'] for match in results['matches']]
            indices = [int(match['id']) for match in results['matches']]
            
            all_distances.append(distances)
            all_indices.append(indices)
        
        return np.array(all_distances), np.array(all_indices)
    
    def save(self, path: str) -> None:
        """Save index metadata (Pinecone index is cloud-hosted)."""
        metadata = self.get_config()
        metadata.update({
            'index_name': self.index_name,
            'environment': self.environment,
        })
        
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, path: str) -> None:
        """Load index metadata and connect to Pinecone."""
        with open(f"{path}.meta", 'rb') as f:
            metadata = pickle.dump(f)
        
        self.dimension = metadata['dimension']
        self.metric = metadata['metric']
        self.n_vectors = metadata['n_vectors']
        self.index_name = metadata['index_name']
        self.environment = metadata['environment']
        
        self._init_pinecone()
        self.index = self.pinecone.Index(self.index_name)
        self.is_built = True