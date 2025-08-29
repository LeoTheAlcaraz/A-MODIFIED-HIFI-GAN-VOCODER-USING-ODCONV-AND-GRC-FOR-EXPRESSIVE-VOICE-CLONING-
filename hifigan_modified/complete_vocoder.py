#!/usr/bin/env python3
"""
Complete Modified HiFi-GAN Vocoder with ODConv and GRC-LoRA
Integrates generator, discriminators, and embedding extractors for expressive voice cloning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .generator import ModifiedHiFiGANGenerator
from .discriminators import HiFiGANDiscriminators
from ..embedding_extractors import EmbeddingExtractor

class ModifiedHiFiGANVocoder(nn.Module):
    """Complete Modified HiFi-GAN Vocoder for expressive voice cloning"""
    
    def __init__(self, input_channels: int = 80, hidden_channels: int = 512,
                 speaker_embedding_dim: int = 192, emotion_embedding_dim: int = 256):
        super().__init__()
        
        # Generator with ODConv and GRC-LoRA
        self.generator = ModifiedHiFiGANGenerator(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            speaker_embedding_dim=speaker_embedding_dim,
            emotion_embedding_dim=emotion_embedding_dim
        )
        
        # Discriminators
        self.discriminators = HiFiGANDiscriminators()
        
        # Embedding extractors
        self.embedding_extractor = EmbeddingExtractor(
            speaker_embedding_dim=speaker_embedding_dim,
            emotion_embedding_dim=emotion_embedding_dim
        )
        
        # Loss weights
        self.fm_weight = 10.0  # Feature matching weight
        self.mel_weight = 45.0  # Mel-spectrogram loss weight
        
    def forward(self, mel_spectrogram: torch.Tensor, 
                speaker_embedding: Optional[torch.Tensor] = None,
                emotion_embedding: Optional[torch.Tensor] = None,
                extract_embeddings: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the complete vocoder
        
        Args:
            mel_spectrogram: Input mel-spectrogram [B, 80, T]
            speaker_embedding: Pre-computed speaker embedding [B, 192] or None
            emotion_embedding: Pre-computed emotion embedding [B, 256] or None
            extract_embeddings: Whether to extract embeddings from input
            
        Returns:
            Dictionary containing outputs and intermediate features
        """
        # Extract embeddings if not provided
        if extract_embeddings and (speaker_embedding is None or emotion_embedding is None):
            extracted_speaker, extracted_emotion = self.embedding_extractor(mel_spectrogram)
            speaker_embedding = speaker_embedding if speaker_embedding is not None else extracted_speaker
            emotion_embedding = emotion_embedding if emotion_embedding is not None else extracted_emotion
        
        # Generate waveform
        generated_waveform = self.generator(
            mel_spectrogram, 
            speaker_embedding, 
            emotion_embedding
        )
        
        return {
            'generated_waveform': generated_waveform,
            'speaker_embedding': speaker_embedding,
            'emotion_embedding': emotion_embedding
        }
    
    def get_discriminator_outputs(self, real_audio: torch.Tensor, 
                                 fake_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get discriminator outputs for training"""
        return self.discriminators(real_audio, fake_audio)
    
    def compute_generator_losses(self, real_audio: torch.Tensor, 
                                fake_audio: torch.Tensor,
                                mel_spectrogram: torch.Tensor,
                                generated_mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute generator losses"""
        
        # Get discriminator outputs
        disc_outputs = self.discriminators(real_audio, fake_audio)
        
        # Adversarial losses
        mpd_losses = []
        msd_losses = []
        
        # MPD losses
        for mpd_real, mpd_fake in zip(disc_outputs['mpd_real'], disc_outputs['mpd_fake']):
            mpd_loss = F.mse_loss(mpd_fake, torch.ones_like(mpd_fake))
            mpd_losses.append(mpd_loss)
        
        # MSD losses
        for msd_real, msd_fake in zip(disc_outputs['msd_real'], disc_outputs['msd_fake']):
            msd_loss = F.mse_loss(msd_fake, torch.ones_like(msd_fake))
            msd_losses.append(msd_loss)
        
        # Feature matching losses
        mpd_fm_losses = []
        msd_fm_losses = []
        
        # MPD feature matching
        for mpd_real, mpd_fake in zip(disc_outputs['mpd_real'], disc_outputs['mpd_fake']):
            fm_loss = F.l1_loss(mpd_fake, mpd_real.detach())
            mpd_fm_losses.append(fm_loss)
        
        # MSD feature matching
        for msd_real, msd_fake in zip(disc_outputs['msd_real'], disc_outputs['msd_fake']):
            fm_loss = F.l1_loss(msd_fake, msd_real.detach())
            msd_fm_losses.append(fm_loss)
        
        # Mel-spectrogram loss
        mel_loss = F.l1_loss(generated_mel, mel_spectrogram)
        
        # Total generator loss
        total_loss = (
            sum(mpd_losses) + sum(msd_losses) +  # Adversarial losses
            self.fm_weight * (sum(mpd_fm_losses) + sum(msd_fm_losses)) +  # Feature matching
            self.mel_weight * mel_loss  # Mel-spectrogram loss
        )
        
        return {
            'total_loss': total_loss,
            'mpd_loss': sum(mpd_losses),
            'msd_loss': sum(msd_losses),
            'mpd_fm_loss': sum(mpd_fm_losses),
            'msd_fm_loss': sum(msd_fm_losses),
            'mel_loss': mel_loss
        }
    
    def compute_discriminator_losses(self, real_audio: torch.Tensor, 
                                   fake_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute discriminator losses"""
        
        # Get discriminator outputs
        disc_outputs = self.discriminators(real_audio, fake_audio)
        
        # MPD losses
        mpd_real_losses = []
        mpd_fake_losses = []
        
        for mpd_real, mpd_fake in zip(disc_outputs['mpd_real'], disc_outputs['mpd_fake']):
            mpd_real_loss = F.mse_loss(mpd_real, torch.ones_like(mpd_real))
            mpd_fake_loss = F.mse_loss(mpd_fake, torch.zeros_like(mpd_fake))
            mpd_real_losses.append(mpd_real_loss)
            mpd_fake_losses.append(mpd_fake_loss)
        
        # MSD losses
        msd_real_losses = []
        msd_fake_losses = []
        
        for msd_real, msd_fake in zip(disc_outputs['msd_real'], disc_outputs['msd_fake']):
            msd_real_loss = F.mse_loss(msd_real, torch.ones_like(msd_real))
            msd_fake_loss = F.mse_loss(msd_fake, torch.zeros_like(msd_fake))
            msd_real_losses.append(msd_real_loss)
            msd_fake_losses.append(msd_fake_loss)
        
        # Total discriminator loss
        total_loss = (
            sum(mpd_real_losses) + sum(mpd_fake_losses) +
            sum(msd_real_losses) + sum(msd_fake_losses)
        )
        
        return {
            'total_loss': total_loss,
            'mpd_real_loss': sum(mpd_real_losses),
            'mpd_fake_loss': sum(mpd_fake_losses),
            'msd_real_loss': sum(msd_real_losses),
            'msd_fake_loss': sum(msd_fake_losses)
        }

class VocoderTrainer:
    """Training wrapper for the modified HiFi-GAN vocoder"""
    
    def __init__(self, vocoder: ModifiedHiFiGANVocoder, 
                 generator_optimizer: torch.optim.Optimizer,
                 discriminator_optimizer: torch.optim.Optimizer,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        self.vocoder = vocoder.to(device)
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.device = device
        
    def train_step(self, mel_spectrogram: torch.Tensor, real_audio: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        
        # Move to device
        mel_spectrogram = mel_spectrogram.to(self.device)
        real_audio = real_audio.to(self.device)
        
        # Generate fake audio
        vocoder_output = self.vocoder(mel_spectrogram)
        fake_audio = vocoder_output['generated_waveform']
        
        # Compute mel-spectrogram of generated audio (simplified)
        # In practice, you'd use a proper mel-spectrogram computation
        generated_mel = mel_spectrogram  # Placeholder
        
        # Train discriminator
        self.discriminator_optimizer.zero_grad()
        disc_losses = self.vocoder.compute_discriminator_losses(real_audio, fake_audio.detach())
        disc_losses['total_loss'].backward()
        self.discriminator_optimizer.step()
        
        # Train generator
        self.generator_optimizer.zero_grad()
        gen_losses = self.vocoder.compute_generator_losses(
            real_audio, fake_audio, mel_spectrogram, generated_mel
        )
        gen_losses['total_loss'].backward()
        self.generator_optimizer.step()
        
        # Return losses
        return {
            'generator_loss': gen_losses['total_loss'].item(),
            'discriminator_loss': disc_losses['total_loss'].item(),
            'mel_loss': gen_losses['mel_loss'].item()
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'vocoder_state_dict': self.vocoder.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict()
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.vocoder.load_state_dict(checkpoint['vocoder_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

if __name__ == "__main__":
    # Test the complete vocoder
    print("Testing Complete Modified HiFi-GAN Vocoder...")
    
    # Create test inputs
    batch_size = 2
    mel_channels = 80
    seq_len = 100
    audio_length = 1000
    
    mel_spec = torch.randn(batch_size, mel_channels, seq_len)
    real_audio = torch.randn(batch_size, 1, audio_length)
    
    # Initialize vocoder
    vocoder = ModifiedHiFiGANVocoder()
    
    # Test forward pass
    outputs = vocoder(mel_spec)
    print(f"Generated waveform shape: {outputs['generated_waveform'].shape}")
    print(f"Speaker embedding shape: {outputs['speaker_embedding'].shape}")
    print(f"Emotion embedding shape: {outputs['emotion_embedding'].shape}")
    
    # Test discriminator outputs
    disc_outputs = vocoder.get_discriminator_outputs(real_audio, outputs['generated_waveform'])
    print(f"Discriminator outputs: {len(disc_outputs)} types")
    
    # Test loss computation
    generated_mel = mel_spec  # Placeholder
    gen_losses = vocoder.compute_generator_losses(real_audio, outputs['generated_waveform'], 
                                                mel_spec, generated_mel)
    disc_losses = vocoder.compute_discriminator_losses(real_audio, outputs['generated_waveform'])
    
    print(f"Generator total loss: {gen_losses['total_loss']:.4f}")
    print(f"Discriminator total loss: {disc_losses['total_loss']:.4f}")
    
    print("✅ Complete Modified HiFi-GAN Vocoder test successful!")
