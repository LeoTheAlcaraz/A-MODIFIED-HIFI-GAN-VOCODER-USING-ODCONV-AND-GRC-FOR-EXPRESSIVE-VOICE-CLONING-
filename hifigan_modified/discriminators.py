#!/usr/bin/env python3
"""
HiFi-GAN Discriminators: Multi-Period and Multi-Scale
Implements the discriminator architecture for adversarial training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator for HiFi-GAN"""
    
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.periods = periods
        self.discriminators = nn.ModuleList([
            Discriminator2D(period) for period in periods
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through all period discriminators"""
        outputs = []
        for disc in self.discriminators:
            output = disc(x)
            outputs.append(output)
        return outputs

class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator for HiFi-GAN"""
    
    def __init__(self, scales: List[int] = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.discriminators = nn.ModuleList([
            Discriminator1D(scale) for scale in scales
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through all scale discriminators"""
        outputs = []
        for disc in self.discriminators:
            output = disc(x)
            outputs.append(output)
        return outputs

class Discriminator2D(nn.Module):
    """2D Discriminator for multi-period evaluation"""
    
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        
        # Reshape 1D audio to 2D for period-based evaluation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 1, (3, 3), padding=(1, 1))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with 2D reshaping"""
        # Reshape 1D audio to 2D for period-based evaluation
        B, C, T = x.shape
        if T % self.period != 0:
            # Pad to make divisible by period
            pad_length = self.period - (T % self.period)
            x = F.pad(x, (0, pad_length))
            T = x.shape[-1]
        
        # Reshape: (B, C, T) -> (B, C, period, T//period)
        x = x.view(B, C, self.period, T // self.period)
        
        # Apply 2D convolutions
        x = self.conv_layers(x)
        
        return x

class Discriminator1D(nn.Module):
    """1D Discriminator for multi-scale evaluation"""
    
    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale
        
        # Downsample input by scale factor
        self.downsample = nn.AvgPool1d(scale, stride=scale)
        
        # 1D convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, 15, padding=7),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, 15, padding=7),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, 15, padding=7),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 256, 15, padding=7),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 1, 15, padding=7)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with downsampling"""
        # Downsample input
        x = self.downsample(x)
        
        # Apply 1D convolutions
        x = self.conv_layers(x)
        
        return x

class HiFiGANDiscriminators(nn.Module):
    """Complete discriminator system for HiFi-GAN"""
    
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
    
    def forward(self, real_audio: torch.Tensor, fake_audio: torch.Tensor) -> dict:
        """
        Forward pass through all discriminators
        
        Args:
            real_audio: Real audio waveform [B, 1, T]
            fake_audio: Generated audio waveform [B, 1, T]
            
        Returns:
            Dictionary containing discriminator outputs
        """
        # MPD outputs
        mpd_real = self.mpd(real_audio)
        mpd_fake = self.mpd(fake_audio)
        
        # MSD outputs
        msd_real = self.msd(real_audio)
        msd_fake = self.msd(fake_audio)
        
        return {
            'mpd_real': mpd_real,
            'mpd_fake': mpd_fake,
            'msd_real': msd_real,
            'msd_fake': msd_fake
        }

if __name__ == "__main__":
    # Test the discriminators
    batch_size = 2
    audio_length = 1000
    
    # Create test inputs
    real_audio = torch.randn(batch_size, 1, audio_length)
    fake_audio = torch.randn(batch_size, 1, audio_length)
    
    # Initialize discriminators
    discriminators = HiFiGANDiscriminators()
    
    # Forward pass
    outputs = discriminators(real_audio, fake_audio)
    
    print(f"Real audio shape: {real_audio.shape}")
    print(f"Fake audio shape: {fake_audio.shape}")
    print(f"MPD real outputs: {len(outputs['mpd_real'])} discriminators")
    print(f"MPD fake outputs: {len(outputs['mpd_fake'])} discriminators")
    print(f"MSD real outputs: {len(outputs['msd_real'])} discriminators")
    print(f"MSD fake outputs: {len(outputs['msd_fake'])} discriminators")
    print("✅ HiFi-GAN Discriminators test successful!")
