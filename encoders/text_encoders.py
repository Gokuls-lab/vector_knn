"""Text encoding implementations."""

from typing import Union, List
import numpy as np
from ..core.base_encoder import BaseEncoder


class SentenceTransformerEncoder(BaseEncoder):
    """Sentence Transformer encoder for text.
    
    Default and recommended for most text classification tasks.
    
    Example:
        >>> encoder = SentenceTransformerEncoder('all-MiniLM-L12-v2')
        >>> embeddings = encoder.encode(['Hello world', 'How are you?'])
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L12-v2',
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        
    def load(self) -> None:
        """Load the model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install: pip install sentence-transformers")
        
        print(f"Loading SentenceTransformer: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.is_loaded = True
        
    def encode(
        self,
        inputs: Union[List[str], str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode text to embeddings."""
        if not self.is_loaded:
            self.load()
        
        if isinstance(inputs, str):
            inputs = [inputs]
        
        embeddings = self.model.encode(
            inputs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            **kwargs
        )
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if not self.is_loaded:
            self.load()
        return self.dimension


class OpenAIEncoder(BaseEncoder):
    """OpenAI embeddings encoder.
    
    Example:
        >>> encoder = OpenAIEncoder(api_key='sk-...')
        >>> embeddings = encoder.encode(['Hello world'])
    """
    
    def __init__(
        self,
        model_name: str = 'text-embedding-ada-002',
        api_key: str = None,
        **kwargs
    ):
        super().__init__(model_name, 'api', **kwargs)
        self.api_key = api_key
        
    def load(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai
        except ImportError:
            raise ImportError("Please install: pip install openai")
        
        if self.api_key:
            openai.api_key = self.api_key
        
        self.model = openai
        self.dimension = 1536  # Ada-002 dimension
        self.is_loaded = True
        
    def encode(
        self,
        inputs: Union[List[str], str],
        batch_size: int = 100,
        show_progress: bool = True,
        normalize: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Encode text using OpenAI API."""
        if not self.is_loaded:
            self.load()
        
        if isinstance(inputs, str):
            inputs = [inputs]
        
        embeddings = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            response = self.model.Embedding.create(
                input=batch,
                model=self.model_name
            )
            batch_embeddings = [item['embedding'] for item in response['data']]
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def get_dimension(self) -> int:
        return self.dimension


class HuggingFaceEncoder(BaseEncoder):
    """Generic HuggingFace transformer encoder.
    
    Example:
        >>> encoder = HuggingFaceEncoder('bert-base-uncased')
        >>> embeddings = encoder.encode(['Hello world'])
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        device: str = 'cpu',
        pooling: str = 'mean',  # 'mean', 'cls', 'max'
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self.pooling = pooling
        
    def load(self) -> None:
        """Load HuggingFace model."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("Please install: pip install transformers torch")
        
        print(f"Loading HuggingFace model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        # Get dimension
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors='pt').to(self.device)
            output = self.model(**dummy)
            self.dimension = output.last_hidden_state.shape[-1]
        
        self.is_loaded = True
        
    def encode(
        self,
        inputs: Union[List[str], str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode text."""
        import torch
        
        if not self.is_loaded:
            self.load()
        
        if isinstance(inputs, str):
            inputs = [inputs]
        
        all_embeddings = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                
                if self.pooling == 'cls':
                    embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.pooling == 'mean':
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                elif self.pooling == 'max':
                    embeddings = outputs.last_hidden_state.max(dim=1)[0]
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def get_dimension(self) -> int:
        if not self.is_loaded:
            self.load()
        return self.dimension