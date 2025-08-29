#!/usr/bin/env python3
"""
Embedding Extractors: ECAPA-TDNN and Emotion2Vec
Implements speaker and emotion embedding extraction for vocoder conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN"""
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 512, 
                 embedding_dim: int = 192, num_speakers: int = 1000):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Initial 1D convolution
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, dilation=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # SE-Res2Blocks with increasing dilation
        self.se_res2_blocks = nn.ModuleList([
            SE_Res2Block(hidden_dim, dilation=2),
            SE_Res2Block(hidden_dim, dilation=3),
            SE_Res2Block(hidden_dim, dilation=4)
        ])
        
        # Channel expansion
        self.channel_expansion = nn.Conv1d(hidden_dim, 3 * hidden_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(3 * hidden_dim)
        
        # Attentive Statistical Pooling
        self.attention = nn.Sequential(
            nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 3 * hidden_dim, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Final projection
        self.final_proj = nn.Linear(3 * hidden_dim, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        
        # Speaker classification head (for training)
        self.speaker_classifier = nn.Linear(embedding_dim, num_speakers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of ECAPA-TDNN
        
        Args:
            x: Input mel-spectrogram [B, 80, T]
            
        Returns:
            embedding: Speaker embedding [B, 192]
            logits: Speaker classification logits [B, num_speakers] (if training)
        """
        # Initial convolution
        x = F.relu(self.bn1(self.input_conv(x)))
        
        # SE-Res2Blocks
        for block in self.se_res2_blocks:
            x = block(x)
        
        # Channel expansion
        x = F.relu(self.bn2(self.channel_expansion(x)))
        
        # Attentive Statistical Pooling
        attention_weights = self.attention(x)
        
        # Apply attention and compute statistics
        attended = x * attention_weights
        mean = torch.mean(attended, dim=2)
        std = torch.std(attended, dim=2)
        
        # Concatenate mean and std
        pooled = torch.cat([mean, std], dim=1)
        
        # Final projection
        embedding = self.final_proj(pooled)
        embedding = self.bn3(embedding)
        
        # L2 normalization
        embedding = F.normalize(embedding, p=2, dim=1)
        
        # Speaker classification (only during training)
        if self.training:
            logits = self.speaker_classifier(embedding)
            return embedding, logits
        else:
            return embedding, None

class SE_Res2Block(nn.Module):
    """Squeeze-and-Excitation Res2Block"""
    
    def __init__(self, channels: int, dilation: int = 1, scale: int = 8):
        super().__init__()
        
        self.channels = channels
        self.scale = scale
        self.dilation = dilation
        
        # Res2Net structure
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)
        
        # Multiple scale convolutions
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(channels // scale, channels // scale, kernel_size=3, 
                     padding=dilation, dilation=dilation)
            for _ in range(scale)
        ])
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
        # Squeeze-and-Excitation
        self.se = SE_Module(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # First 1x1 conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Res2Net multi-scale processing
        xs = torch.chunk(x, self.scale, dim=1)
        ys = []
        for i, conv in enumerate(self.scale_convs):
            if i == 0:
                ys.append(xs[i])
            else:
                ys.append(conv(xs[i] + ys[-1]))
        
        # Concatenate and second 1x1 conv
        x = torch.cat(ys, dim=1)
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Squeeze-and-Excitation
        x = self.se(x)
        
        return x + residual

class SE_Module(nn.Module):
    """Squeeze-and-Excitation module"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class Emotion2Vec(nn.Module):
    """Emotion2Vec: Self-supervised speech emotion representation model"""
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 512, 
                 embedding_dim: int = 256, num_emotions: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # 1D CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotions)
        )
        
        # Frame-level emotion projection
        self.frame_projection = nn.Linear(hidden_dim, embedding_dim)
        
        # Utterance-level emotion projection
        self.utterance_projection = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Emotion2Vec
        
        Args:
            x: Input mel-spectrogram [B, 80, T]
            
        Returns:
            frame_embeddings: Frame-level emotion embeddings [B, T, 256]
            utterance_embedding: Utterance-level emotion embedding [B, 256]
            emotion_logits: Emotion classification logits [B, num_emotions] (if training)
        """
        # Feature extraction
        features = self.feature_extractor(x)  # [B, H, T]
        
        # Transpose for transformer
        features = features.transpose(1, 2)  # [B, T, H]
        
        # Transformer encoding
        encoded = self.transformer(features)  # [B, T, H]
        
        # Frame-level embeddings
        frame_embeddings = self.frame_projection(encoded)  # [B, T, 256]
        
        # Utterance-level embedding (mean pooling)
        utterance_embedding = torch.mean(encoded, dim=1)  # [B, H]
        utterance_embedding = self.utterance_projection(utterance_embedding)  # [B, 256]
        
        # L2 normalization
        utterance_embedding = F.normalize(utterance_embedding, p=2, dim=1)
        
        # Emotion classification (only during training)
        if self.training:
            emotion_logits = self.emotion_classifier(utterance_embedding)
            return frame_embeddings, utterance_embedding, emotion_logits
        else:
            return frame_embeddings, utterance_embedding, None

class EmbeddingExtractor(nn.Module):
    """Combined embedding extractor for both speaker and emotion"""
    
    def __init__(self, speaker_embedding_dim: int = 192, emotion_embedding_dim: int = 256):
        super().__init__()
        
        self.speaker_extractor = ECAPA_TDNN(embedding_dim=speaker_embedding_dim)
        self.emotion_extractor = Emotion2Vec(embedding_dim=emotion_embedding_dim)
        
    def forward(self, mel_spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract both speaker and emotion embeddings
        
        Args:
            mel_spectrogram: Input mel-spectrogram [B, 80, T]
            
        Returns:
            speaker_embedding: Speaker embedding [B, 192]
            emotion_embedding: Emotion embedding [B, 256]
        """
        # Extract speaker embedding
        speaker_embedding, _ = self.speaker_extractor(mel_spectrogram)
        
        # Extract emotion embedding
        _, emotion_embedding, _ = self.emotion_extractor(mel_spectrogram)
        
        return speaker_embedding, emotion_embedding

if __name__ == "__main__":
    # Test the embedding extractors
    batch_size = 2
    mel_channels = 80
    seq_len = 100
    
    # Create test input
    mel_spec = torch.randn(batch_size, mel_channels, seq_len)
    
    # Test ECAPA-TDNN
    print("Testing ECAPA-TDNN...")
    ecapa = ECAPA_TDNN()
    speaker_emb, _ = ecapa(mel_spec)
    print(f"Speaker embedding shape: {speaker_emb.shape}")
    
    # Test Emotion2Vec
    print("Testing Emotion2Vec...")
    emotion2vec = Emotion2Vec()
    frame_emb, utt_emb, _ = emotion2vec(mel_spec)
    print(f"Frame embeddings shape: {frame_emb.shape}")
    print(f"Utterance embedding shape: {utt_emb.shape}")
    
    # Test combined extractor
    print("Testing combined extractor...")
    extractor = EmbeddingExtractor()
    spk_emb, emo_emb = extractor(mel_spec)
    print(f"Combined speaker embedding: {spk_emb.shape}")
    print(f"Combined emotion embedding: {emo_emb.shape}")
    
    print("✅ All embedding extractors test successful!")
