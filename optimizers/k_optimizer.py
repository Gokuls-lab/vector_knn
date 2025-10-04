"""K-value optimization using Bayesian Optimization and Grid Search."""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os

try:
    from bayes_opt import BayesianOptimization
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False


class KOptimizer:
    """
    Bayesian Optimization and Grid Search for finding optimal K value.
    
    Supports multiple optimization strategies:
    - Bayesian Optimization (efficient, smart exploration)
    - Grid Search (exhaustive, simple)
    - Random Search (fast, good baseline)
    
    Parameters:
        classifier: Fitted classifier instance
        k_range: Tuple of (min_k, max_k) to search
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
        cv: Number of cross-validation folds (None for single validation set)
        
    Example:
        >>> # Basic usage
        >>> clf = FAISSKNNClassifier()
        >>> clf.fit(X_train, y_train)
        >>> 
        >>> optimizer = KOptimizer(clf, k_range=(1, 50), metric='f1')
        >>> best_k = optimizer.optimize(X_val, y_val)
        >>> 
        >>> # Update classifier
        >>> clf.config.k_neighbors = best_k
        >>> 
        >>> # Or use auto-optimization
        >>> clf.fit_with_auto_k(X_train, y_train, X_val, y_val)
    """
    
    def __init__(
        self,
        classifier,
        k_range: Tuple[int, int] = (1, 50),
        metric: str = 'f1',
        cv: Optional[int] = None
    ):
        """Initialize the optimizer."""
        if not hasattr(classifier, 'is_fitted_') or not classifier.is_fitted_:
            raise ValueError("Classifier must be fitted before optimization")
        
        self.classifier = classifier
        self.k_range = k_range
        self.metric = metric
        self.cv = cv
        
        # Results storage
        self.optimization_history_: List[Dict] = []
        self.best_k_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.best_params_: Optional[Dict] = None
        
        # Validate metric
        valid_metrics = ['f1', 'accuracy', 'precision', 'recall']
        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of {valid_metrics}, got '{metric}'")
    
    def _calculate_score(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """Calculate score based on selected metric."""
        if self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'f1':
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        elif self.metric == 'precision':
            return precision_score(y_true, y_pred, average='weighted', zero_division=0)
        elif self.metric == 'recall':
            return recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    def _evaluate_k(
        self, 
        k: Union[float, int], 
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> float:
        """Evaluate a specific K value.
        
        Args:
            k: K value to evaluate
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Score for the given K
        """
        k = int(round(k))
        k = max(1, min(k, len(self.classifier.labels_) - 1))  # Ensure valid K
        
        # Get predictions
        y_pred = self.classifier.predict(X_val, k=k)
        
        # Calculate score
        score = self._calculate_score(y_val, y_pred)
        
        # Store history
        self.optimization_history_.append({
            'k': k,
            'score': score,
            'metric': self.metric
        })
        
        return score
    
    def bayesian_optimize(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        init_points: int = 5,
        n_iter: int = 15,
        verbose: bool = True
    ) -> int:
        """Find optimal K value using Bayesian Optimization.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            init_points: Number of random exploration points
            n_iter: Number of Bayesian optimization iterations
            verbose: Whether to print optimization progress
            
        Returns:
            Optimal K value
        """
        if not BAYESIAN_OPT_AVAILABLE:
            raise ImportError(
                "Bayesian optimization requires bayesian-optimization package. "
                "Install with: pip install bayesian-optimization"
            )
        
        print("\n" + "=" * 60)
        print("🔍 BAYESIAN OPTIMIZATION FOR K VALUE")
        print("=" * 60)
        print(f"K range: {self.k_range}")
        print(f"Metric: {self.metric}")
        print(f"Validation samples: {len(y_val)}")
        print(f"Random exploration: {init_points} points")
        print(f"Bayesian iterations: {n_iter}")
        print("=" * 60)
        
        # Clear history
        self.optimization_history_ = []
        
        # Define objective function
        def objective(k):
            return self._evaluate_k(k, X_val, y_val)
        
        # Setup optimizer
        optimizer = BayesianOptimization(
            f=objective,
            pbounds={'k': self.k_range},
            random_state=getattr(self.classifier.config, 'random_state', 42),
            verbose=2 if verbose else 0
        )
        
        # Run optimization
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        # Extract best K
        self.best_k_ = int(round(optimizer.max['params']['k']))
        self.best_score_ = optimizer.max['target']
        self.best_params_ = {'k': self.best_k_, 'score': self.best_score_}
        
        print("\n" + "=" * 60)
        print("🎯 BAYESIAN OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"Best K: {self.best_k_}")
        print(f"Best {self.metric}: {self.best_score_:.4f}")
        print(f"Total evaluations: {len(self.optimization_history_)}")
        print("=" * 60)
        
        return self.best_k_
    
    def grid_search(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        k_values: Optional[List[int]] = None,
        verbose: bool = True
    ) -> int:
        """Find optimal K using grid search.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            k_values: List of K values to test (auto-generated if None)
            verbose: Whether to print progress
            
        Returns:
            Optimal K value
        """
        if k_values is None:
            # Generate smart grid
            k_values = self._generate_smart_grid()
        
        # Filter to valid range
        k_values = [k for k in k_values if self.k_range[0] <= k <= self.k_range[1]]
        
        print("\n" + "=" * 60)
        print("🔍 GRID SEARCH FOR K VALUE")
        print("=" * 60)
        print(f"Testing K values: {k_values}")
        print(f"Metric: {self.metric}")
        print(f"Validation samples: {len(y_val)}")
        print("=" * 60)
        
        # Clear history
        self.optimization_history_ = []
        
        results = []
        for i, k in enumerate(k_values):
            score = self._evaluate_k(k, X_val, y_val)
            results.append({'k': k, 'score': score})
            
            if verbose:
                print(f"[{i+1}/{len(k_values)}] K={k:3d}: {self.metric}={score:.4f}")
        
        # Find best
        best_result = max(results, key=lambda x: x['score'])
        self.best_k_ = best_result['k']
        self.best_score_ = best_result['score']
        self.best_params_ = best_result
        
        print("\n" + "=" * 60)
        print("🎯 GRID SEARCH RESULTS")
        print("=" * 60)
        print(f"Best K: {self.best_k_}")
        print(f"Best {self.metric}: {self.best_score_:.4f}")
        print("=" * 60)
        
        return self.best_k_
    
    def random_search(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 20,
        verbose: bool = True
    ) -> int:
        """Find optimal K using random search.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of random trials
            verbose: Whether to print progress
            
        Returns:
            Optimal K value
        """
        print("\n" + "=" * 60)
        print("🔍 RANDOM SEARCH FOR K VALUE")
        print("=" * 60)
        print(f"K range: {self.k_range}")
        print(f"Trials: {n_trials}")
        print(f"Metric: {self.metric}")
        print("=" * 60)
        
        # Clear history
        self.optimization_history_ = []
        
        # Generate random K values
        np.random.seed(getattr(self.classifier.config, 'random_state', 42))
        k_values = np.random.randint(self.k_range[0], self.k_range[1] + 1, size=n_trials)
        
        results = []
        for i, k in enumerate(k_values):
            score = self._evaluate_k(k, X_val, y_val)
            results.append({'k': k, 'score': score})
            
            if verbose:
                print(f"[{i+1}/{n_trials}] K={k:3d}: {self.metric}={score:.4f}")
        
        # Find best
        best_result = max(results, key=lambda x: x['score'])
        self.best_k_ = best_result['k']
        self.best_score_ = best_result['score']
        self.best_params_ = best_result
        
        print("\n" + "=" * 60)
        print("🎯 RANDOM SEARCH RESULTS")
        print("=" * 60)
        print(f"Best K: {self.best_k_}")
        print(f"Best {self.metric}: {self.best_score_:.4f}")
        print("=" * 60)
        
        return self.best_k_
    
    def optimize(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = 'bayesian',
        **kwargs
    ) -> int:
        """Optimize K using specified method.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            method: Optimization method ('bayesian', 'grid', 'random')
            **kwargs: Method-specific parameters
            
        Returns:
            Optimal K value
        """
        if method == 'bayesian':
            return self.bayesian_optimize(X_val, y_val, **kwargs)
        elif method == 'grid':
            return self.grid_search(X_val, y_val, **kwargs)
        elif method == 'random':
            return self.random_search(X_val, y_val, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bayesian', 'grid', or 'random'")
    
    def _generate_smart_grid(self) -> List[int]:
        """Generate a smart grid of K values to test."""
        min_k, max_k = self.k_range
        
        # Denser sampling for small K values, sparser for large K
        grid = []
        
        # Fine-grained for small K (1-10)
        grid.extend(range(1, min(11, max_k + 1)))
        
        # Medium-grained for medium K (11-30)
        if max_k > 10:
            grid.extend(range(15, min(31, max_k + 1), 5))
        
        # Coarse-grained for large K (30+)
        if max_k > 30:
            grid.extend(range(40, max_k + 1, 10))
        
        # Remove duplicates and sort
        grid = sorted(list(set([k for k in grid if min_k <= k <= max_k])))
        
        return grid
    
    def plot_optimization_history(
        self, 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 5)
    ) -> None:
        """Plot optimization history.
        
        Args:
            save_path: Path to save the plot (optional)
            figsize: Figure size
        """
        if not self.optimization_history_:
            print("⚠️  No optimization history available. Run optimize() first.")
            return
        
        df = pd.DataFrame(self.optimization_history_)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Score over iterations
        axes[0].plot(range(len(df)), df['score'], 'b-o', linewidth=2, markersize=6)
        axes[0].axhline(
            y=self.best_score_, 
            color='r', 
            linestyle='--', 
            linewidth=2,
            label=f'Best {self.metric}: {self.best_score_:.4f}'
        )
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel(f'{self.metric.capitalize()} Score', fontsize=12)
        axes[0].set_title('Optimization Progress', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: K vs Score (scatter)
        scatter = axes[1].scatter(
            df['k'], df['score'], 
            c=range(len(df)), 
            cmap='viridis', 
            s=100, 
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
        axes[1].scatter(
            self.best_k_, self.best_score_, 
            color='red', 
            s=400, 
            marker='*', 
            edgecolors='black', 
            linewidths=2, 
            label=f'Best K={self.best_k_}',
            zorder=10
        )
        axes[1].set_xlabel('K Value', fontsize=12)
        axes[1].set_ylabel(f'{self.metric.capitalize()} Score', fontsize=12)
        axes[1].set_title('K Value vs Performance', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        plt.colorbar(scatter, ax=axes[1], label='Iteration')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Optimization plot saved to {save_path}")
        
        plt.show()
    
    def plot_k_comparison(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """Plot comparison of all tested K values.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.optimization_history_:
            print("⚠️  No optimization history available. Run optimize() first.")
            return
        
        df = pd.DataFrame(self.optimization_history_)
        
        # Group by K and take mean (in case K was tested multiple times)
        df_grouped = df.groupby('k')['score'].agg(['mean', 'std', 'count']).reset_index()
        df_grouped = df_grouped.sort_values('k')
        
        plt.figure(figsize=figsize)
        
        # Plot mean scores
        plt.plot(df_grouped['k'], df_grouped['mean'], 'b-o', linewidth=2, markersize=8, label='Score')
        
        # Add error bars if multiple measurements
        has_std = df_grouped['count'].max() > 1
        if has_std:
            plt.fill_between(
                df_grouped['k'],
                df_grouped['mean'] - df_grouped['std'],
                df_grouped['mean'] + df_grouped['std'],
                alpha=0.2,
                color='blue',
                label='±1 std'
            )
        
        # Highlight best K
        plt.axvline(
            x=self.best_k_, 
            color='r', 
            linestyle='--', 
            linewidth=2, 
            label=f'Optimal K={self.best_k_}'
        )
        
        plt.xlabel('K Value', fontsize=12)
        plt.ylabel(f'{self.metric.capitalize()} Score', fontsize=12)
        plt.title(f'K Value Comparison ({self.metric})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ K comparison plot saved to {save_path}")
        
        plt.show()
    
    def get_optimization_results(self) -> pd.DataFrame:
        """Get optimization results as DataFrame.
        
        Returns:
            DataFrame with K values and scores, sorted by score
        """
        if not self.optimization_history_:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.optimization_history_)
        
        # Group by K and aggregate
        df_grouped = df.groupby('k').agg({
            'score': ['mean', 'std', 'count', 'min', 'max']
        }).reset_index()
        
        df_grouped.columns = ['k', 'mean_score', 'std_score', 'count', 'min_score', 'max_score']
        df_grouped['is_best'] = df_grouped['k'] == self.best_k_
        df_grouped = df_grouped.sort_values('mean_score', ascending=False).reset_index(drop=True)
        
        return df_grouped
    
    def print_summary(self) -> None:
        """Print optimization summary."""
        if self.best_k_ is None:
            print("⚠️  No optimization performed yet.")
            return
        
        print("\n" + "=" * 60)
        print("📊 OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Metric optimized: {self.metric}")
        print(f"K range searched: {self.k_range}")
        print(f"Total evaluations: {len(self.optimization_history_)}")
        print(f"\n🎯 Best K: {self.best_k_}")
        print(f"   Best {self.metric}: {self.best_score_:.4f}")
        
        # Show top 5 K values
        results_df = self.get_optimization_results()
        if len(results_df) > 0:
            print(f"\n📈 Top 5 K values:")
            print(results_df[['k', 'mean_score', 'count']].head(5).to_string(index=False))
        
        print("=" * 60)


def auto_optimize_k(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    val_size: float = 0.2,
    method: str = 'bayesian',
    k_range: Tuple[int, int] = (1, 50),
    metric: str = 'f1',
    update_classifier: bool = True,
    verbose: bool = True,
    **optimizer_kwargs
) -> Tuple[int, float]:
    """
    Auto-optimize K value for a classifier.
    
    This is a convenience function that handles validation split automatically.
    
    Args:
        classifier: Fitted or unfitted classifier
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional, will split from train if None)
        y_val: Validation labels (optional)
        val_size: Validation split size if X_val is None
        method: Optimization method ('bayesian', 'grid', 'random')
        k_range: K range to search
        metric: Metric to optimize
        update_classifier: Whether to update classifier with best K
        verbose: Print progress
        **optimizer_kwargs: Additional arguments for optimizer
        
    Returns:
        Tuple of (best_k, best_score)
        
    Example:
        >>> clf = FAISSKNNClassifier()
        >>> best_k, score = auto_optimize_k(clf, X_train, y_train, method='bayesian')
        >>> # Classifier is automatically updated with best K
        >>> predictions = clf.predict(X_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Split validation set if not provided
    if X_val is None or y_val is None:
        if verbose:
            print(f"Splitting {val_size*100:.0f}% of training data for validation...")
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=getattr(classifier.config, 'random_state', 42),
            stratify=y_train
        )
    else:
        X_train_split = X_train
        y_train_split = y_train
    
    # Fit classifier if not already fitted
    if not hasattr(classifier, 'is_fitted_') or not classifier.is_fitted_:
        if verbose:
            print("Fitting classifier on training data...")
        classifier.fit(X_train_split, y_train_split)
    
    # Create optimizer
    optimizer = KOptimizer(classifier, k_range=k_range, metric=metric)
    
    # Optimize
    if verbose:
        print(f"\n🔍 Auto-optimizing K using {method} search...")
    
    best_k = optimizer.optimize(X_val, y_val, method=method, verbose=verbose, **optimizer_kwargs)
    best_score = optimizer.best_score_
    
    # Update classifier
    if update_classifier:
        classifier.config.k_neighbors = best_k
        if verbose:
            print(f"\n✓ Classifier updated with optimal K={best_k}")
    
    return best_k, best_score