"""Core classifier implementation."""

from typing import Union, List, Dict, Any, Optional, Tuple
import faiss
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


from .base_encoder import BaseEncoder
from .base_index import BaseIndex
from ..config import Config
from ..utils.data_utils import prepare_input_data
from ..utils.metrics import calculate_metrics, print_classification_report
from ..utils.visualization import plot_confusion_matrix


class VectorKNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Highly Customizable Vector Space K-Nearest Neighbors Classifier.
    
    This classifier allows full customization of:
    - Encoder: Any embedding model (text, image, audio, video, multimodal)
    - Index: Any vector search backend (FAISS, Pinecone, Milvus, etc.)
    - Distance metric: Cosine, Euclidean, Dot product, etc.
    - K value: Number of neighbors
    
    Parameters:
        encoder: Encoder instance for generating embeddings
        index: Index instance for similarity search
        config: Configuration object
        k_neighbors: Number of neighbors (overrides config)
        
    Example:
        >>> # Basic usage with defaults
        >>> from vector_knn import FAISSKNNClassifier
        >>> clf = FAISSKNNClassifier()
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        
        >>> # Advanced: Full customization
        >>> from vector_knn import VectorKNNClassifier, CLIPImageEncoder, PineconeIndex
        >>> encoder = CLIPImageEncoder(model_name='openai/clip-vit-base-patch32')
        >>> index = PineconeIndex(dimension=512, metric='cosine', api_key='...')
        >>> clf = VectorKNNClassifier(encoder=encoder, index=index, k_neighbors=10)
        >>> clf.fit(image_paths, labels)
        >>> predictions = clf.predict(new_images)
    """
    
    def __init__(
        self,
        encoder: Optional[BaseEncoder] = None,
        index: Optional[BaseIndex] = None,
        config: Optional[Config] = None,
        k_neighbors: Optional[int] = None,
        **kwargs  # Accept additional kwargs to support loading
    ):
        """Initialize the classifier."""
        self.config = config or Config()
        self.encoder = encoder
        self.index = index
        
        if k_neighbors is not None:
            self.config.k_neighbors = k_neighbors
        
        # State
        self.labels_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted_: bool = False
        self._encoder_provided = encoder is not None
        self._index_provided = index is not None
        
    def _initialize_encoder(self) -> None:
        """Initialize encoder if not provided."""
        if self.encoder is None:
            # Use default text encoder
            from ..encoders.text_encoders import SentenceTransformerEncoder
            print("No encoder provided. Using default: SentenceTransformerEncoder")
            self.encoder = SentenceTransformerEncoder(
                model_name='all-MiniLM-L12-v2',
                device=self.config.device
            )
        
        if not self.encoder.is_loaded:
            self.encoder.load()
    
    def _initialize_index(self, dimension: int) -> None:
        """Initialize index if not provided."""
        if self.index is None:
            # Use default FAISS index
            from ..indexes.faiss_index import FAISSIndex
            print("No index provided. Using default: FAISSIndex")
            self.index = FAISSIndex(
                index_type=self.config.index_type,
                dimension=dimension,
                metric=self.config.distance_metric
            )
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.Series, np.ndarray, List],
        feature_names: Optional[List[str]] = None
    ) -> 'VectorKNNClassifier':
        """Fit the classifier on training data.
        
        Args:
            X: Training data (format depends on encoder)
            y: Training labels
            feature_names: Feature names (optional)
            
        Returns:
            self: Fitted classifier
        """
        print("\n" + "=" * 60)
        print("FITTING VECTOR KNN CLASSIFIER")
        print("=" * 60)
        
        # Initialize encoder
        self._initialize_encoder()
        
        # Prepare data
        X_prepared, self.feature_names_ = prepare_input_data(
            X, feature_names, encoder_type=self.encoder.__class__.__name__
        )
        
        self.labels_ = np.array(y)
        self.classes_ = np.unique(self.labels_)
        
        print(f"Training samples: {len(self.labels_)}")
        print(f"Number of classes: {len(self.classes_)}")
        print(f"Encoder: {self.encoder}")
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        embeddings = self.encoder.encode(
            X_prepared,
            batch_size=self.config.batch_size,
            show_progress=self.config.show_progress,
            normalize=self.config.normalize_embeddings
        )
        
        dimension = embeddings.shape[1]
        print(f"Embedding dimension: {dimension}")
        
        # Initialize and build index
        self._initialize_index(dimension)
        
        print(f"Index: {self.index}")
        print("\nBuilding search index...")
        self.index.build(embeddings)
        
        self.is_fitted_ = True
        print("\n✓ Model fitted successfully!")
        print("=" * 60)
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        k: Optional[int] = None
    ) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Input data
            k: Number of neighbors (uses config default if None)
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        k = k or self.config.k_neighbors
        
        # Prepare data
        X_prepared, _ = prepare_input_data(
            X, self.feature_names_, encoder_type=self.encoder.__class__.__name__
        )
        
        # Generate embeddings
        query_embeddings = self.encoder.encode(
            X_prepared,
            batch_size=self.config.batch_size,
            show_progress=self.config.show_progress and len(X_prepared) > 100,
            normalize=self.config.normalize_embeddings
        )
        
        # Search
        _, indices = self.index.search(query_embeddings, k)
        
        # Majority voting
        predictions = []
        for neighbor_indices in indices:
            neighbor_labels = self.labels_[neighbor_indices]
            predicted = np.bincount(neighbor_labels).argmax()
            predictions.append(predicted)
        
        return np.array(predictions)
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        k: Optional[int] = None
    ) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input data
            k: Number of neighbors
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        k = k or self.config.k_neighbors
        
        X_prepared, _ = prepare_input_data(
            X, self.feature_names_, encoder_type=self.encoder.__class__.__name__
        )
        
        query_embeddings = self.encoder.encode(
            X_prepared,
            batch_size=self.config.batch_size,
            show_progress=self.config.show_progress and len(X_prepared) > 100,
            normalize=self.config.normalize_embeddings
        )
        
        _, indices = self.index.search(query_embeddings, k)
        
        probas = []
        for neighbor_indices in indices:
            neighbor_labels = self.labels_[neighbor_indices]
            class_counts = np.bincount(neighbor_labels, minlength=len(self.classes_))
            class_proba = class_counts / k
            probas.append(class_proba)
        
        return np.array(probas)
    
    def predict_with_details(
        self,
        X: Union[pd.DataFrame, np.ndarray, List, Any],
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Predict with detailed information about neighbors.
        
        Args:
            X: Input data (single sample or batch)
            k: Number of neighbors
            
        Returns:
            List of dictionaries with prediction details
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        k = k or self.config.k_neighbors
        
        # Handle single sample
        if not isinstance(X, (list, pd.DataFrame, np.ndarray)):
            X = [X]
            single_sample = True
        else:
            single_sample = False
        
        X_prepared, _ = prepare_input_data(
            X, self.feature_names_, encoder_type=self.encoder.__class__.__name__
        )
        
        query_embeddings = self.encoder.encode(
            X_prepared,
            batch_size=self.config.batch_size,
            show_progress=False,
            normalize=self.config.normalize_embeddings
        )
        
        distances, indices = self.index.search(query_embeddings, k)
        
        results = []
        for i, (neighbor_indices, neighbor_distances) in enumerate(zip(indices, distances)):
            neighbor_labels = self.labels_[neighbor_indices]
            predicted = np.bincount(neighbor_labels).argmax()
            confidence = np.sum(neighbor_labels == predicted) / k
            
            result = {
                'prediction': predicted,
                'confidence': confidence,
                'neighbor_labels': neighbor_labels.tolist(),
                'neighbor_distances': neighbor_distances.tolist(),
                'neighbor_indices': neighbor_indices.tolist(),
                'k': k
            }
            results.append(result)
        
        return results[0] if single_sample else results
    
    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.Series, np.ndarray, List],
        k: Optional[int] = None,
        metric: str = 'accuracy'
    ) -> float:
        """Compute score on test data.
        
        Args:
            X: Test data
            y: True labels
            k: Number of neighbors
            metric: Metric to use
            
        Returns:
            Score value
        """
        y_pred = self.predict(X, k=k)
        y_true = np.array(y)
        
        metrics = calculate_metrics(y_true, y_pred)
        
        if metric not in metrics:
            raise ValueError(f"Unknown metric: {metric}")
        
        return metrics[metric]
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.Series, np.ndarray, List],
        k: Optional[int] = None,
        verbose: bool = True,
        plot: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """Comprehensive evaluation.
        
        Args:
            X: Test data
            y: True labels
            k: Number of neighbors
            verbose: Print detailed report
            plot: Plot confusion matrix
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X, k=k)
        y_true = np.array(y)
        
        metrics = calculate_metrics(y_true, y_pred)
        
        if verbose:
            print_classification_report(y_true, y_pred)
        
        if plot:
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "confusion_matrix.png")
            plot_confusion_matrix(y_true, y_pred, save_path=save_path)
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save the model.
        
        Args:
            path: Directory path to save model
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted model.")
        
        os.makedirs(path, exist_ok=True)
        
        # Save index
        index_path = os.path.join(path, "index")
        self.index.save(index_path)
        print(f"✓ Index saved to {index_path}")
        
        # Save metadata
        metadata = {
            'labels': self.labels_,
            'classes': self.classes_,
            'feature_names': self.feature_names_,
            'config': self.config,
            'encoder_class': self.encoder.__class__.__name__,
            'encoder_config': {
                'model_name': self.encoder.model_name,
                'device': self.encoder.device,
            },
            'index_class': self.index.__class__.__name__,
            'index_config': self.index.get_config(),
        }
        
        metadata_path = os.path.join(path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Metadata saved to {metadata_path}")
        
        print(f"\n🎉 Model successfully saved to '{path}'")
    
    @classmethod
    def load(cls, path: str, encoder: Optional[BaseEncoder] = None) -> 'VectorKNNClassifier':
        """Load a saved model.
        
        Args:
            path: Directory path containing saved model
            encoder: Encoder instance (will be recreated if None)
            
        Returns:
            Loaded classifier
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory not found: {path}")
        
        # Load metadata
        metadata_path = os.path.join(path, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Recreate encoder if not provided
        if encoder is None:
            encoder_class_name = metadata['encoder_class']
            encoder_config = metadata['encoder_config']
            
            # Import encoder class dynamically
            from .. import encoders
            encoder_class = getattr(encoders, encoder_class_name)
            encoder = encoder_class(**encoder_config)
            encoder.load()
        
        # Recreate index
        index_class_name = metadata['index_class']
        index_config = metadata['index_config']
        
        from .. import indexes
        index_class = getattr(indexes, index_class_name)
        index = index_class(**{k: v for k, v in index_config.items() if k != 'n_vectors'})
        
        # Load index
        index_path = os.path.join(path, "index")
        index.load(index_path)
        
        # Create classifier with loaded components
        instance = cls(
            config=metadata['config'],
            encoder=encoder,
            index=index
        )
        
        # Set state
        instance.labels_ = metadata['labels']
        instance.classes_ = metadata['classes']
        instance.feature_names_ = metadata['feature_names']
        instance.is_fitted_ = True
        
        print(f"🎉 Model successfully loaded from '{path}'")
        
        return instance
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters (sklearn compatibility)."""
        return {
            'encoder': self.encoder,
            'index': self.index,
            'config': self.config,
            'k_neighbors': self.config.k_neighbors,
        }
    
    def set_params(self, **params) -> 'VectorKNNClassifier':
        """Set parameters (sklearn compatibility)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
        return self
    # Add to VectorKNNClassifier class

    def fit_with_auto_k(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        y: Union[pd.Series, np.ndarray, List],
        X_val: Optional[Union[pd.DataFrame, np.ndarray, List]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray, List]] = None,
        val_size: float = 0.2,
        k_range: Tuple[int, int] = (1, 50),
        optimization_method: str = 'bayesian',
        optimization_metric: str = 'f1',
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
        **optimizer_kwargs
    ) -> 'VectorKNNClassifier':
        """
        Fit classifier and automatically optimize K value.
        
        This is a convenience method that combines fitting and K optimization.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional, will split from train if None)
            y_val: Validation labels (optional)
            val_size: Validation split size if X_val is None
            k_range: K range to search
            optimization_method: 'bayesian', 'grid', or 'random'
            optimization_metric: Metric to optimize ('f1', 'accuracy', etc.)
            feature_names: Feature names (optional)
            verbose: Print progress
            **optimizer_kwargs: Additional arguments for optimizer
            
        Returns:
            self: Fitted classifier with optimized K
            
        Example:
            >>> clf = FAISSKNNClassifier()
            >>> clf.fit_with_auto_k(X_train, y_train, optimization_method='bayesian')
            >>> # K is automatically optimized and set
            >>> predictions = clf.predict(X_test)
        """
        from sklearn.model_selection import train_test_split
        from ..optimizers import KOptimizer
        
        # Split validation set if not provided
        if X_val is None or y_val is None:
            if verbose:
                print(f"📊 Splitting {val_size*100:.0f}% of training data for K optimization...")
            
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X, y,
                test_size=val_size,
                random_state=self.config.random_state,
                stratify=y if len(np.unique(y)) < len(y) * 0.5 else None
            )
        else:
            X_train_split = X
            y_train_split = y
            X_val_split = X_val
            y_val_split = y_val
        
        # Fit on training data
        if verbose:
            print("🔨 Fitting classifier on training data...")
        self.fit(X_train_split, y_train_split, feature_names=feature_names)
        
        # Optimize K
        if verbose:
            print(f"\n🎯 Optimizing K value using {optimization_method} search...")
        
        optimizer = KOptimizer(self, k_range=k_range, metric=optimization_metric)
        best_k = optimizer.optimize(
            X_val_split, 
            y_val_split, 
            method=optimization_method,
            verbose=verbose,
            **optimizer_kwargs
        )
        
        # Update K
        self.config.k_neighbors = best_k
        
        if verbose:
            print(f"\n✓ Training complete! Optimal K={best_k} ({optimization_metric}={optimizer.best_score_:.4f})")
        
        # Store optimizer for later inspection
        self.optimizer_ = optimizer
        
        return self

    def optimize_k(
        self,
        X_val: Union[pd.DataFrame, np.ndarray, List],
        y_val: Union[pd.Series, np.ndarray, List],
        k_range: Tuple[int, int] = (1, 50),
        method: str = 'bayesian',
        metric: str = 'f1',
        update_k: bool = True,
        verbose: bool = True,
        plot: bool = True,
        **optimizer_kwargs
    ) -> Tuple[int, float]:
        """
        Optimize K value on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            k_range: K range to search
            method: Optimization method
            metric: Metric to optimize
            update_k: Whether to update classifier's K value
            verbose: Print progress
            plot: Plot optimization results
            **optimizer_kwargs: Additional optimizer arguments
            
        Returns:
            Tuple of (best_k, best_score)
            
        Example:
            >>> clf.fit(X_train, y_train)
            >>> best_k, score = clf.optimize_k(X_val, y_val, method='grid')
        """
        from ..optimizers import KOptimizer
        
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before K optimization. Call fit() first.")
        
        optimizer = KOptimizer(self, k_range=k_range, metric=metric)
        best_k = optimizer.optimize(X_val, y_val, method=method, verbose=verbose, **optimizer_kwargs)
        
        if update_k:
            self.config.k_neighbors = best_k
            if verbose:
                print(f"\n✓ Classifier updated with optimal K={best_k}")
        
        if plot:
            optimizer.plot_optimization_history()
            optimizer.plot_k_comparison()
        
        # Store optimizer
        self.optimizer_ = optimizer
        
        return best_k, optimizer.best_score_

    def get_optimization_results(self) -> Optional[pd.DataFrame]:
        """Get K optimization results if optimization was performed.
        
        Returns:
            DataFrame with optimization results or None
        """
        if hasattr(self, 'optimizer_'):
            return self.optimizer_.get_optimization_results()
        else:
            print("⚠️  No optimization performed yet. Use fit_with_auto_k() or optimize_k() first.")
            return None



# import faiss
# import numpy as np
# import pandas as pd
# from typing import Union, List, Optional, Dict
from sklearn.metrics import silhouette_score, davies_bouldin_score

# from .base_encoder import BaseEncoder
# from .base_index import BaseIndex
# from ..config import Config
# from ..utils.data_utils import prepare_input_data


class VectorKNNClusterer:
    """
    Unsupervised Vector KNN Clusterer using FAISS.
    
    - Converts raw input (text, image, etc.) into embeddings using an encoder
    - Clusters embeddings with FAISS KMeans or sklearn DBSCAN
    - Evaluates clusters with unsupervised metrics (silhouette, DB index)
    
    Example:
        >>> from vector_knn import FAISSKNNClusterer
        >>> clusterer = FAISSKNNClusterer(k_clusters=5, method="kmeans")
        >>> clusterer.fit(text_data)
        >>> labels = clusterer.get_labels()
        >>> clusterer.evaluate()
    """

    def __init__(
        self,
        encoder: Optional[BaseEncoder] = None,
        index: Optional[BaseIndex] = None,
        config: Optional[Config] = None,
        k_clusters: int = 5,
        method: str = "kmeans",  # 'kmeans' | 'dbscan'
        **kwargs
    ):
        self.encoder = encoder
        self.index = index
        self.config = config or Config()
        self.k_clusters = k_clusters
        self.method = method

        self.is_fitted_ = False
        self.labels_: Optional[np.ndarray] = None
        self.centroids_: Optional[np.ndarray] = None
        self.embeddings_: Optional[np.ndarray] = None

    def _initialize_encoder(self) -> None:
        """Initialize encoder if not provided (default: SentenceTransformer)."""
        if self.encoder is None:
            from ..encoders.text_encoders import SentenceTransformerEncoder
            print("No encoder provided. Using default: SentenceTransformerEncoder")
            self.encoder = SentenceTransformerEncoder(
                model_name="all-MiniLM-L12-v2",
                device=self.config.device
            )
        if not self.encoder.is_loaded:
            self.encoder.load()

    def fit(self, X: Union[pd.DataFrame, np.ndarray, List], feature_names: Optional[List[str]] = None):
        """Fit clustering on raw unlabeled data (text, images, etc.)."""
        print("\n" + "=" * 60)
        print("FITTING VECTOR KNN CLUSTERER (Unsupervised)")
        print("=" * 60)

        # Initialize encoder
        self._initialize_encoder()

        # Prepare data
        X_prepared, _ = prepare_input_data(
            X, feature_names, encoder_type=self.encoder.__class__.__name__
        )

        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings_ = self.encoder.encode(
            X_prepared,
            batch_size=self.config.batch_size,
            show_progress=self.config.show_progress,
            normalize=self.config.normalize_embeddings
        )
        self.embeddings_ = np.array(self.embeddings_).astype("float32")
        d = self.embeddings_.shape[1]

        # Run clustering
        if self.method == "kmeans":
            print(f"Running FAISS KMeans with k={self.k_clusters}")
            kmeans = faiss.Kmeans(d, self.k_clusters, niter=20, verbose=True)
            kmeans.train(self.embeddings_)

            index = faiss.IndexFlatL2(d)
            index.add(kmeans.centroids)
            _, labels = index.search(self.embeddings_, 1)

            self.labels_ = labels.flatten()
            self.centroids_ = kmeans.centroids

        elif self.method == "dbscan":
            from sklearn.cluster import DBSCAN
            print("Running DBSCAN on embeddings...")
            db = DBSCAN(eps=1.5, min_samples=5, metric="euclidean")
            self.labels_ = db.fit_predict(self.embeddings_)

        else:
            raise ValueError("Unknown clustering method: choose 'kmeans' or 'dbscan'")

        self.is_fitted_ = True
        print("✓ Clustering complete!")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """Assign clusters to new data (only works for KMeans)."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.method != "kmeans":
            raise ValueError("Prediction only supported for kmeans.")

        # Encode
        X_prepared, _ = prepare_input_data(
            X, None, encoder_type=self.encoder.__class__.__name__
        )
        query_embeddings = self.encoder.encode(X_prepared)
        query_embeddings = np.array(query_embeddings).astype("float32")

        d = query_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(self.centroids_)
        _, labels = index.search(query_embeddings, 1)
        return labels.flatten()

    def get_labels(self) -> np.ndarray:
        """Return cluster assignments for training data."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")
        return self.labels_

    def evaluate(self) -> Dict[str, float]:
        """Compute clustering metrics (unsupervised)."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")

        scores = {}
        if len(set(self.labels_)) > 1:
            scores["silhouette"] = silhouette_score(self.embeddings_, self.labels_)
            scores["davies_bouldin"] = davies_bouldin_score(self.embeddings_, self.labels_)
        else:
            scores["silhouette"] = None
            scores["davies_bouldin"] = None

        print("\nClustering Metrics:")
        for k, v in scores.items():
            print(f"{k}: {v}")
        return scores
