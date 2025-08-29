import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config
import numpy as np
import os

class Emotion2Vec(nn.Module):
    """
    Emotion2Vec: Emotion-aware speech representation learning
    Based on Ma et al., 2024 with modifications for HiFi-GAN integration
    """
    
    def __init__(self, 
                 hidden_size=768, 
                 num_emotions=8,
                 num_layers=12,
                 dropout=0.1):
        super().__init__()
        
        # Wav2Vec2 backbone for speech feature extraction
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_emotions)
        )
        
        # Emotion embedding projection
        self.emotion_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        
        # Emotion labels mapping
        self.emotion_labels = {
            0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry',
            4: 'fearful', 5: 'disgusted', 6: 'surprised', 7: 'excited'
        }
        
    def forward(self, audio_input, return_emotion_logits=False):
        """
        Forward pass for emotion embedding extraction
        
        Args:
            audio_input: Audio tensor of shape (batch_size, sequence_length)
            return_emotion_logits: Whether to return emotion classification logits
            
        Returns:
            emotion_embedding: Emotion-aware embedding of shape (batch_size, hidden_size//2)
            emotion_logits: (optional) Emotion classification logits
        """
        
        # Extract features using Wav2Vec2
        with torch.no_grad():
            wav2vec2_output = self.wav2vec2(audio_input)
            hidden_states = wav2vec2_output.last_hidden_state  # (B, T, H)
        
        # Global average pooling over time dimension
        pooled_features = torch.mean(hidden_states, dim=1)  # (B, H)
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(pooled_features)
        
        # Emotion-aware embedding projection
        emotion_embedding = self.emotion_projection(pooled_features)
        
        if return_emotion_logits:
            return emotion_embedding, emotion_logits
        else:
            return emotion_embedding
    
    def get_emotion_label(self, emotion_logits):
        """Convert emotion logits to emotion labels"""
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        emotion_indices = torch.argmax(emotion_probs, dim=-1)
        
        labels = []
        for idx in emotion_indices:
            labels.append(self.emotion_labels[idx.item()])
        
        return labels, emotion_probs

def load_emotion2vec_model(model_path=None, device='cuda'):
    """
    Load Emotion2Vec model
    
    Args:
        model_path: Path to pretrained model (if None, loads default)
        device: Device to load model on
        
    Returns:
        Emotion2Vec model instance
    """
    model = Emotion2Vec()
    
    if model_path and os.path.exists(model_path):
        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded Emotion2Vec from {model_path}")
    else:
        print("Using Emotion2Vec with default initialization")
    
    model = model.to(device)
    model.eval()
    
    return model

# Emotion embedding utilities
def extract_emotion_embeddings(audio_batch, model, device='cuda'):
    """
    Extract emotion embeddings from audio batch
    
    Args:
        audio_batch: Audio tensor of shape (batch_size, sequence_length)
        model: Emotion2Vec model
        device: Device to run inference on
        
    Returns:
        emotion_embeddings: Emotion embeddings
        emotion_labels: Predicted emotion labels
    """
    model.eval()
    with torch.no_grad():
        audio_batch = audio_batch.to(device)
        emotion_embeddings, emotion_logits = model(audio_batch, return_emotion_logits=True)
        emotion_labels, _ = model.get_emotion_label(emotion_logits)
    
    return emotion_embeddings, emotion_labels 