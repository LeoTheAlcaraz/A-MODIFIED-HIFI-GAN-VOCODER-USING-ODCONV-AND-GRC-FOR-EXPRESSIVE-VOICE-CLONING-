"""
Modified HiFi-GAN vocoder package with ODConv and GRC-LoRA
"""

from .generator import ModifiedHiFiGANGenerator
from .discriminators import HiFiGANDiscriminators
from .complete_vocoder import ModifiedHiFiGANVocoder, VocoderTrainer

__all__ = [
    'ModifiedHiFiGANGenerator',
    'HiFiGANDiscriminators',
    'ModifiedHiFiGANVocoder',
    'VocoderTrainer'
]
