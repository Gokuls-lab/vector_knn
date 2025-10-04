"""Pre-configured classifier presets for easy usage."""

from typing import Optional
from ..core.base_classifier import VectorKNNClassifier
from ..core.base_classifier  import VectorKNNClusterer
from ..core.base_index import BaseIndex
from ..encoders.text_encoders import SentenceTransformerEncoder
from ..encoders.image_encoders import CLIPImageEncoder, ResNetEncoder, ConvNeXtEncoder
from ..indexes.faiss_index import FAISSIndex
from ..indexes.pinecone_index import PineconeIndex
from ..config import Config


# ============================================================================
# TEXT CLASSIFIERS
# ============================================================================

class FAISSKNNClassifier(VectorKNNClassifier):
    """
    Ready-to-use FAISS-based text classifier with sensible defaults.
    
    Perfect for getting started quickly with text classification.
    
    Example:
        >>> from vector_knn import FAISSKNNClassifier
        >>> clf = FAISSKNNClassifier(k_neighbors=5)
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        embedding_model: str = 'all-MiniLM-L12-v2',
        device: str = 'cpu',
        metric: str = 'cosine',
        config: Optional[Config] = None,
        encoder: Optional[SentenceTransformerEncoder] = None,
        index: Optional[BaseIndex] = None,
        index_type: str = 'Flat',  # 'Flat', 'IVF', 'HNSW'
        nlist: int = 100,  # For IVF
        **kwargs
    ):
        """Initialize FAISS KNN classifier with text encoder."""
        if config is None and 'config' not in kwargs:
            config = Config(k_neighbors=k_neighbors, distance_metric=metric, device=device, index_type=index_type, nlist=nlist)
        
        if encoder is None and 'encoder' not in kwargs:
            encoder = SentenceTransformerEncoder(model_name=embedding_model, device=device)
        
        if index is None and 'index' not in kwargs:
            index = None  # Will be created automatically after encoding
        
        if config is not None:
            kwargs['config'] = config
        if encoder is not None:
            kwargs['encoder'] = encoder
        if index is not None:
            kwargs['index'] = index
            
        super().__init__(**kwargs)

class FAISSKNNClusterer(VectorKNNClusterer):
    """
    Ready-to-use FAISS-based unsupervised clusterer with sensible defaults.

    Perfect for quick clustering of text (or other data via encoders).

    Example:
        >>> from vector_knn import FAISSKNNClusterer
        >>> clusterer = FAISSKNNClusterer(k_clusters=5, method="kmeans")
        >>> clusterer.fit(texts)   # unlabeled data
        >>> labels = clusterer.get_labels()
        >>> metrics = clusterer.evaluate()
    """

    def __init__(
        self,
        k_clusters: int = 5,
        method: str = "kmeans",  # 'kmeans' or 'dbscan'
        embedding_model: str = "all-MiniLM-L12-v2",
        device: str = "cpu",
        metric: str = "cosine",
        config: Optional[Config] = None,
        encoder: Optional[SentenceTransformerEncoder] = None,
        index: Optional[BaseIndex] = None,
        **kwargs
    ):
        """Initialize FAISS KNN clusterer with text encoder."""

        # Config
        if config is None and "config" not in kwargs:
            config = Config(
                k_neighbors=k_clusters,
                device=device
            )

        # Encoder
        if encoder is None and "encoder" not in kwargs:
            encoder = SentenceTransformerEncoder(
                model_name=embedding_model,
                device=device
            )

        # Index (optional, created if None)
        if index is None and "index" not in kwargs:
            index = None

        if config is not None:
            kwargs["config"] = config
        if encoder is not None:
            kwargs["encoder"] = encoder
        if index is not None:
            kwargs["index"] = index

        kwargs["k_clusters"] = k_clusters
        kwargs["method"] = method

        super().__init__(**kwargs)


class TextFAISSClassifier(FAISSKNNClassifier):
    """Alias for FAISSKNNClassifier for clarity."""
    pass


class ConvNeXtClassifier(VectorKNNClassifier):
    """
    Ready-to-use ConvNeXt-based image classifier.
    
    Uses ConvNeXt architecture for high-performance image classification.
    
    Example:
        >>> from vector_knn import ConvNeXtClassifier
        >>> clf = ConvNeXtClassifier(k_neighbors=5)
        >>> clf.fit(image_paths, labels)
        >>> predictions = clf.predict(new_images)
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        model_name: str = 'facebook/convnext-base-224',
        device: str = 'cpu',
        metric: str = 'cosine',
        config: Optional[Config] = None,
        encoder: Optional[ConvNeXtEncoder] = None,
        index: Optional[BaseIndex] = None,
        **kwargs
    ):
        """Initialize ConvNeXt classifier."""
        if config is None and 'config' not in kwargs:
            config = Config(k_neighbors=k_neighbors, distance_metric=metric, device=device)
        
        if encoder is None and 'encoder' not in kwargs:
            encoder = ConvNeXtEncoder(model_name=model_name, device=device)
        
        if index is None and 'index' not in kwargs:
            index = None  # Will be created automatically after encoding
        
        if config is not None:
            kwargs['config'] = config
        if encoder is not None:
            kwargs['encoder'] = encoder
        if index is not None:
            kwargs['index'] = index
            
        super().__init__(**kwargs)


class PineconeKNNClassifier(VectorKNNClassifier):
    """
    Ready-to-use Pinecone-based text classifier.
    
    Uses cloud-hosted Pinecone for scalable vector search.
    
    Example:
        >>> from vector_knn import PineconeKNNClassifier
        >>> clf = PineconeKNNClassifier(
        ...     api_key='your-api-key',
        ...     environment='us-west1-gcp',
        ...     index_name='my-classifier'
        ... )
        >>> clf.fit(X_train, y_train)
    """
    
    def __init__(
        self,
        api_key: str,
        environment: str = 'us-west1-gcp',
        index_name: str = 'vector-knn',
        k_neighbors: int = 5,
        embedding_model: str = 'all-MiniLM-L12-v2',
        device: str = 'cpu',
        metric: str = 'cosine',
        config: Optional[Config] = None,
        **kwargs
    ):
        """Initialize Pinecone KNN classifier."""
        if config is None:
            config = Config(k_neighbors=k_neighbors, distance_metric=metric, device=device)
        
        encoder = SentenceTransformerEncoder(model_name=embedding_model, device=device)
        
        # Note: dimension will be set after first encoding
        # For now, use a placeholder that will be updated
        index = None  # Will be created in fit() with correct dimension
        
        # Store Pinecone config for later
        self._pinecone_config = {
            'api_key': api_key,
            'environment': environment,
            'index_name': index_name
        }
        
        super().__init__(encoder=encoder, index=index, config=config, **kwargs)
    
    def _initialize_index(self, dimension: int) -> None:
        """Override to create Pinecone index."""
        if self.index is None:
            self.index = PineconeIndex(
                dimension=dimension,
                metric=self.config.distance_metric,
                **self._pinecone_config
            )


# ============================================================================
# IMAGE CLASSIFIERS
# ============================================================================

class ImageFAISSClassifier(VectorKNNClassifier):
    """
    Ready-to-use FAISS-based image classifier using CLIP.
    
    Example:
        >>> from vector_knn import ImageFAISSClassifier
        >>> clf = ImageFAISSClassifier()
        >>> clf.fit(image_paths_train, y_train)
        >>> predictions = clf.predict(image_paths_test)
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        embedding_model: str = 'openai/clip-vit-base-patch32',
        device: str = 'cpu',
        metric: str = 'cosine',
        config: Optional[Config] = None,
        **kwargs
    ):
        """Initialize image classifier with CLIP encoder."""
        if config is None:
            config = Config(k_neighbors=k_neighbors, distance_metric=metric, device=device)
        
        encoder = CLIPImageEncoder(model_name=embedding_model, device=device)
        index = None
        
        super().__init__(encoder=encoder, index=index, config=config, **kwargs)


class ImageResNetClassifier(VectorKNNClassifier):
    """
    Image classifier using ResNet encoder.
    
    Example:
        >>> from vector_knn import ImageResNetClassifier
        >>> clf = ImageResNetClassifier(embedding_model='resnet50')
        >>> clf.fit(images, labels)
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        embedding_model: str = 'resnet50',
        device: str = 'cpu',
        metric: str = 'cosine',
        config: Optional[Config] = None,
        **kwargs
    ):
        """Initialize ResNet-based image classifier."""
        if config is None:
            config = Config(k_neighbors=k_neighbors, distance_metric=metric, device=device)
        
        encoder = ResNetEncoder(model_name=embedding_model, device=device)
        index = None
        
        super().__init__(encoder=encoder, index=index, config=config, **kwargs)


# ============================================================================
# PLACEHOLDERS FOR OTHER MODALITIES
# ============================================================================

class AudioFAISSClassifier(VectorKNNClassifier):
    """Audio classifier - to be implemented"""
    pass


class VideoFAISSClassifier(VectorKNNClassifier):
    """Video classifier - to be implemented"""
    pass


class MultimodalFAISSClassifier(VectorKNNClassifier):
    """Multimodal classifier - to be implemented"""
    pass


# ============================================================================
# OTHER INDEX BACKENDS
# ============================================================================

class MilvusKNNClassifier(VectorKNNClassifier):
    """Milvus-based classifier - to be implemented"""
    pass


class AnnoyKNNClassifier(VectorKNNClassifier):
    """Annoy-based classifier - to be implemented"""
    pass


class ChromaKNNClassifier(VectorKNNClassifier):
    """ChromaDB-based classifier - to be implemented"""
    pass