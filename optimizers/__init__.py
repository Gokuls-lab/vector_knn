"""Optimization utilities for Vector KNN Classifier."""

from .k_optimizer import KOptimizer, auto_optimize_k
from .hyperparameter_optimizer import HyperparameterOptimizer

def auto_optimize_hyperparameters(
    classifier,
    X_val,
    y_val,
    param_space=None,
    method='bayesian',
    metric='f1',
    cv=None,
    verbose=True,
    update_classifier=True,
    n_iter=30,
    init_points=5
):
    """
    Automatically optimize hyperparameters for a classifier.
    
    Args:
        classifier: The classifier to optimize
        X_val: Validation features
        y_val: Validation labels
        param_space: Dict of parameters to optimize. If None, uses default space
        method: Optimization method ('bayesian' or 'grid')
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
        cv: Number of cross-validation folds
        verbose: Whether to print progress
        update_classifier: Whether to update classifier with best params
        
    Returns:
        Tuple of (best_params, best_score)
    """
    if param_space is None:
        param_space = {
            'k': (1, 50),
            'metric': ['l2', 'cosine', 'inner_product'],
            'weight': ['uniform', 'distance']
        }
        
    optimizer = HyperparameterOptimizer(
        classifier=classifier,
        param_space=param_space,
        metric=metric,
        cv=cv,
        n_iter=n_iter,
        init_points=init_points
    )
    
    best_params = optimizer.optimize(X_val, y_val, method=method, n_iter=n_iter, verbose=verbose, init_points=init_points)
    
    if update_classifier:
        for param, value in best_params.items():
            if param == 'k':
                classifier.config.k_neighbors = value
            elif param == 'metric':
                classifier.config.metric = value
            elif param == 'weight':
                classifier.config.weight = value
                
    if verbose:
        print(f"Best parameters found: {best_params}")
        
    return best_params

__all__ = [
    'KOptimizer',
    'auto_optimize_k',
    'HyperparameterOptimizer',
    'auto_optimize_hyperparameters'
]