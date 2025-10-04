"""Image encoding implementations."""

from typing import Union, List
import numpy as np
from PIL import Image
from ..core.base_encoder import BaseEncoder


class CLIPImageEncoder(BaseEncoder):
    """CLIP image encoder.
    
    Example:
        >>> encoder = CLIPImageEncoder('openai/clip-vit-base-patch32')
        >>> embeddings = encoder.encode(['path/to/image1.jpg', 'path/to/image2.jpg'])
    """
    
    def __init__(
        self,
        model_name: str = 'openai/clip-vit-base-patch32',
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        
    def load(self) -> None:
        """Load CLIP model."""
        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            raise ImportError("Please install: pip install transformers pillow")
        
        print(f"Loading CLIP model: {self.model_name}")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.dimension = self.model.config.projection_dim
        self.is_loaded = True
        
    def _load_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """Load image from path or PIL Image."""
        if isinstance(image_input, str):
            return Image.open(image_input).convert('RGB')
        return image_input
    
    def encode(
        self,
        inputs: Union[List[Union[str, Image.Image]], Union[str, Image.Image]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode images."""
        import torch
        
        if not self.is_loaded:
            self.load()
        
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        # Load images
        images = [self._load_image(img) for img in inputs]
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            inputs_processed = self.processor(
                images=batch,
                return_tensors='pt',
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs_processed)
                
                if normalize:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(image_features.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_dimension(self) -> int:
        if not self.is_loaded:
            self.load()
        return self.dimension


class ResNetEncoder(BaseEncoder):
    """ResNet image encoder.
    
    Example:
        >>> encoder = ResNetEncoder('resnet50')
        >>> embeddings = encoder.encode(['image1.jpg', 'image2.jpg'])
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        
    def load(self) -> None:
        """Load ResNet model."""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
        except ImportError:
            raise ImportError("Please install: pip install torch torchvision pillow")
        
        print(f"Loading ResNet model: {self.model_name}")
        
        # Load model
        model_fn = getattr(models, self.model_name)
        self.model = model_fn(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove FC layer
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.dimension = 2048 if self.model_name in ['resnet50', 'resnet101', 'resnet152'] else 512
        self.is_loaded = True
        
    def encode(
        self,
        inputs: Union[List[Union[str, Image.Image]], Union[str, Image.Image]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode images."""
        import torch
        
        if not self.is_loaded:
            self.load()
        
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        # Load and transform images
        images = []
        for img in inputs:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            images.append(self.transform(img))
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i + batch_size]).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model(batch).squeeze()
                all_embeddings.append(embeddings.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def get_dimension(self) -> int:
        if not self.is_loaded:
            self.load()
        return self.dimension


class ConvNeXtEncoder(BaseEncoder):
    """ConvNeXt encoder for images.
    
    Example:
        >>> encoder = ConvNeXtEncoder('facebook/convnext-base-224')
        >>> embeddings = encoder.encode(['image1.jpg', 'image2.jpg'])
    """
    
    def __init__(
        self,
        model_name: str = 'facebook/convnext-base-224',
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        
    def load(self) -> None:
        """Load ConvNeXt model."""
        try:
            from transformers import ConvNextImageProcessor, ConvNextModel
        except ImportError:
            raise ImportError("Please install: pip install transformers pillow")
        
        print(f"Loading ConvNeXt model: {self.model_name}")
        self.processor = ConvNextImageProcessor.from_pretrained(self.model_name)
        self.model = ConvNextModel.from_pretrained(self.model_name).to(self.device)
        self.dimension = self.model.config.hidden_sizes[-1]  # Use the last layer's dimension
        self.is_loaded = True
        
    def encode(
        self,
        inputs: Union[List[Union[str, Image.Image]], Union[str, Image.Image]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode images."""
        import torch
        
        if not self.is_loaded:
            self.load()
        
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        # Load images
        images = []
        for img in inputs:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            images.append(img)
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            inputs_processed = self.processor(batch, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs_processed)
                # Use pooled output for embeddings
                embeddings = outputs.pooler_output
                
                if normalize:
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_dimension(self) -> int:
        if not self.is_loaded:
            self.load()
        return self.dimension


class ViTEncoder(BaseEncoder):
    """Vision Transformer encoder."""
    
    def __init__(
        self,
        model_name: str = 'google/vit-base-patch16-224',
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        
    def load(self) -> None:
        """Load ViT model."""
        try:
            from transformers import ViTImageProcessor, ViTModel
        except ImportError:
            raise ImportError("Please install: pip install transformers pillow")
        
        print(f"Loading ViT model: {self.model_name}")
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTModel.from_pretrained(self.model_name).to(self.device)
        self.dimension = self.model.config.hidden_size
        self.is_loaded = True
        
    def encode(
        self,
        inputs: Union[List[Union[str, Image.Image]], Union[str, Image.Image]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode images."""
        import torch
        
        if not self.is_loaded:
            self.load()
        
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        # Load images
        images = []
        for img in inputs:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            images.append(img)
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            inputs_processed = self.processor(batch, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs_processed)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                
                if normalize:
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_dimension(self) -> int:
        if not self.is_loaded:
            self.load()
        return self.dimension