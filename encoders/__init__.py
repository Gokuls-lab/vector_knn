"""Encoder implementations for different modalities."""

from .text_encoders import (
    SentenceTransformerEncoder,
    OpenAIEncoder,
    HuggingFaceEncoder,
)

from .image_encoders import (
    CLIPImageEncoder,
    ResNetEncoder,
    ViTEncoder,
)

# Placeholder for other modalities (implement as needed)
class Wav2VecEncoder:
    """Audio encoder - to be implemented"""
    pass

class WhisperEncoder:
    """Audio encoder - to be implemented"""
    pass

class VideoMAEEncoder:
    """Video encoder - to be implemented"""
    pass

class CLIPMultimodalEncoder:
    """Multimodal encoder - to be implemented"""
    pass

__all__ = [
    # Text
    'SentenceTransformerEncoder',
    'OpenAIEncoder',
    'HuggingFaceEncoder',
    # Image
    'CLIPImageEncoder',
    'ResNetEncoder',
    'ViTEncoder',
    # Audio (placeholders)
    'Wav2VecEncoder',
    'WhisperEncoder',
    # Video (placeholders)
    'VideoMAEEncoder',
    # Multimodal (placeholders)
    'CLIPMultimodalEncoder',
]