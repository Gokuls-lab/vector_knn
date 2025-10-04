"""Hyperparameter optimization using Bayesian Optimization and Grid Search."""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import warnings

try:
    from bayes_opt import BayesianOptimization
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False


class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization for vector KNN models.
    
    Supports optimization of multiple parameters including:
    - k_neighbors (number of neighbors)
    - distance metric (l2, cosine, inner_product)
    - weight function (uniform, distance)
    
    Parameters:
        classifier: Base classifier instance to optimize
        param_space: Dict of parameters and their ranges/options
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
        cv: Number of cross-validation folds (None for single validation set)
        
    Example:
        >>> # Define parameter space
        >>> params = {
        >>>     'k': (1, 50),
        >>>     'metric': ['l2', 'cosine', 'inner_product'],
        >>>     'weight': ['uniform', 'distance']
        >>> }
        >>> 
        >>> optimizer = HyperparameterOptimizer(
        >>>     classifier=clf,
        >>>     param_space=params,
        >>>     metric='f1'
        >>> )
        >>> best_params = optimizer.optimize(X_val, y_val)
    """
    
    def __init__(
        self,
        classifier,
        param_space: Dict[str, Union[Tuple[int, int], List[str]]],
        metric: str = 'f1',
        n_iter: int = 25,
        cv: Optional[int] = None,
        init_points: int = 5
    ):
        self.classifier = classifier
        self.param_space = param_space
        self.metric = metric
        self.cv = cv
        self.n_iter = n_iter
        self.init_points = init_points
        self._validate_params()
        
    def _validate_params(self):
        """Validate parameter space configuration."""
        valid_metrics = ['l2', 'cosine', 'inner_product']
        valid_weights = ['uniform', 'distance']
        
        if 'metric' in self.param_space:
            for metric in self.param_space['metric']:
                if metric not in valid_metrics:
                    raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
                    
        if 'weight' in self.param_space:
            for weight in self.param_space['weight']:
                if weight not in valid_weights:
                    raise ValueError(f"Invalid weight: {weight}. Must be one of {valid_weights}")

    def _get_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate score based on specified metric."""
        if self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'f1':
            return f1_score(y_true, y_pred)
        elif self.metric == 'precision':
            return precision_score(y_true, y_pred)
        elif self.metric == 'recall':
            return recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _objective(self, **params) -> float:
        """Objective function for optimization."""
        # Update classifier with current parameters
        for param, value in params.items():
            if param == 'k':
                self.classifier.config.k_neighbors = int(value)
            elif param == 'metric':
                self.classifier.config.metric = value
            elif param == 'weight':
                self.classifier.config.weight = value

        if self.cv is None:
            try:
                y_pred = self.classifier.predict(self.X_val)
                score = self._get_score(self.y_val, y_pred)
            except Exception as e:
                warnings.warn(f"Error during prediction: {str(e)}")
                return 0.0
        else:
            cv_scores = []
            kf = KFold(n_splits=self.cv, shuffle=True)
            
            for train_idx, val_idx in kf.split(self.X_val):
                X_fold, y_fold = self.X_val[val_idx], self.y_val[val_idx]
                try:
                    y_pred = self.classifier.predict(X_fold)
                    score = self._get_score(y_fold, y_pred)
                    cv_scores.append(score)
                except Exception as e:
                    warnings.warn(f"Error during CV: {str(e)}")
                    return 0.0
            
            score = np.mean(cv_scores)
            
        return score

    def optimize(self, X_val, y_val, method='bayesian', n_iter=30, verbose=True, init_points=None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using specified method.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            method: Optimization method ('bayesian' or 'grid')
            n_iter: Number of iterations for Bayesian optimization
            init_points: Number of initial random points for Bayesian optimization
            
        Returns:
            Dict of best parameters
        """
        self.X_val = X_val
        self.y_val = y_val
        
        if method == 'bayesian':
            if not BAYESIAN_OPT_AVAILABLE:
                raise ImportError("Bayesian optimization requires bayes_opt package")

            # Use instance init_points if not provided to method
            if init_points is None:
                init_points = self.init_points
                
            # Convert param space to format expected by BayesianOptimization
            pbounds = {}
            categorical_params = {}
            
            for param, space in self.param_space.items():
                if isinstance(space, tuple):
                    pbounds[param] = space
                else:
                    # Handle categorical variables by mapping to integers
                    categorical_params[param] = space
                    pbounds[f"{param}_idx"] = (0, len(space) - 1)
            
            def _objective_wrapper(**kwargs):
                params = kwargs.copy()
                # Convert categorical indices back to actual values
                for param, options in categorical_params.items():
                    idx = int(params.pop(f"{param}_idx"))
                    params[param] = options[idx]
                return self._objective(**params)
            
            optimizer = BayesianOptimization(
                f=_objective_wrapper,
                pbounds=pbounds,
                random_state=42,
                verbose=2 if verbose else 0
            )
            
            # Adjust n_iter based on init_points
            adjusted_n_iter = n_iter - init_points
            if adjusted_n_iter < 0:
                init_points = n_iter
                adjusted_n_iter = 0
            
            optimizer.maximize(
                init_points=init_points,
                n_iter=adjusted_n_iter
            )
            
            # Convert best params back to original format
            best_params = optimizer.max['params']
            for param, options in categorical_params.items():
                idx = int(best_params.pop(f"{param}_idx"))
                best_params[param] = options[idx]
                
            # Ensure k is integer
            if 'k' in best_params:
                best_params['k'] = int(best_params['k'])
                
            return best_params
            
        elif method == 'grid':
            # Implement grid search
            best_score = -np.inf
            best_params = {}
            
            # Generate grid of parameters
            param_grid = []
            for param, space in self.param_space.items():
                if isinstance(space, tuple):
                    values = np.linspace(space[0], space[1], num=10, dtype=int)
                else:
                    values = space
                param_grid.append((param, values))
            
            # Try all combinations
            from itertools import product
            for values in product(*[v for _, v in param_grid]):
                params = {param: value for (param, _), value in zip(param_grid, values)}
                score = self._objective(**params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            return best_params
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
