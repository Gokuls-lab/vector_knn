"""Data preparation utilities."""

from typing import Union, List, Tuple, Optional
import pandas as pd
import numpy as np


def row_to_text(row: Union[pd.Series, dict]) -> str:
    """Convert a data row to text format.
    
    Args:
        row: Data row
        
    Returns:
        Formatted text string
    """
    if isinstance(row, pd.Series):
        row = row.to_dict()
    return " ".join([f"{k}:{v}" for k, v in row.items()])


def prepare_input_data(
    X: Union[pd.DataFrame, np.ndarray, List],
    feature_names: Optional[List[str]] = None,
    encoder_type: str = 'SentenceTransformerEncoder'
) -> Tuple[List, Optional[List[str]]]:
    """Prepare input data based on encoder type.
    
    Args:
        X: Input data
        feature_names: Feature names
        encoder_type: Type of encoder being used
        
    Returns:
        Tuple of (prepared_data, feature_names)
    """
    # Text encoders expect strings or list of strings
    if 'Text' in encoder_type or 'Sentence' in encoder_type or 'OpenAI' in encoder_type:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            return [row_to_text(row) for _, row in X.iterrows()], feature_names
        elif isinstance(X, np.ndarray):
            if feature_names is None:
                raise ValueError("feature_names required for numpy arrays with text encoders")
            df = pd.DataFrame(X, columns=feature_names)
            return [row_to_text(row) for _, row in df.iterrows()], feature_names
        elif isinstance(X, list):
            return X, feature_names
            
    # Image/Audio/Video encoders expect file paths or arrays
    elif 'Image' in encoder_type or 'Audio' in encoder_type or 'Video' in encoder_type:
        if isinstance(X, list):
            return X, feature_names
        elif isinstance(X, pd.DataFrame):
            # Assume first column contains paths
            return X.iloc[:, 0].tolist(), X.columns.tolist()
        elif isinstance(X, np.ndarray):
            return X.tolist(), feature_names
    
    # Default: pass through
    return X, feature_names