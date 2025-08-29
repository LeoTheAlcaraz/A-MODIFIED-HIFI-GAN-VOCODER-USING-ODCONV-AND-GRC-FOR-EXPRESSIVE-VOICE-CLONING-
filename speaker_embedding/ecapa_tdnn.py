import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier
import os

class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN speaker encoder implementation
    Based on SpeechBrain's implementation with modifications for HiFi-GAN integration
    """
    
    def __init__(self, 
                 input_size=80,
                 hidden_size=1024,
                 embedding_size=192,
                 num_layers=3,
                 dropout=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        # TDNN layers
        self.tdnn_layers = nn.ModuleList([
            nn.Conv1d(input_size, hidden_size, kernel_size=5, dilation=1),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=2),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=3),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=1),
            nn.Conv1d(hidden_size, hidden_size * 3, kernel_size=1, dilation=1)
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_size * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, hidden_size * 3, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Final embedding projection
        self.embedding_layer = nn.Linear(hidden_size * 3, embedding_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(len(self.tdnn_layers))
        ])
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_size)
            
        Returns:
            Speaker embedding of shape (batch_size, embedding_size)
        """
        # Transpose for conv1d: (batch_size, input_size, time_steps)
        x = x.transpose(1, 2)
        
        # Apply TDNN layers
        for i, layer in enumerate(self.tdnn_layers):
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Apply layer normalization
            x = x.transpose(1, 2)  # (batch_size, time_steps, hidden_size)
            x = self.layer_norms[i](x)
            x = x.transpose(1, 2)  # Back to (batch_size, hidden_size, time_steps)
        
        # Apply attention
        attention_weights = self.attention(x)
        attended_features = torch.sum(x * attention_weights, dim=2)  # (batch_size, hidden_size * 3)
        
        # Project to embedding
        embedding = self.embedding_layer(attended_features)
        
        # L2 normalization
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

def load_speaker_encoder(model_path=None, device='cuda'):
    """
    Load speaker encoder (ECAPA-TDNN)
    
    Args:
        model_path: Path to pretrained model (if None, uses SpeechBrain pretrained)
        device: Device to load model on
        
    Returns:
        Speaker encoder model
    """
    if model_path and os.path.exists(model_path):
        # Load custom trained model
        model = ECAPA_TDNN()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded ECAPA-TDNN from {model_path}")
    else:
        # Use SpeechBrain pretrained model
        print("Using SpeechBrain pretrained ECAPA-TDNN")
        model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")
    
    model = model.to(device)
    model.eval()
    
    return model

def extract_speaker_embeddings(audio_batch, model, device='cuda'):
    """
    Extract speaker embeddings from audio batch
    
    Args:
        audio_batch: Audio tensor of shape (batch_size, sequence_length)
        model: Speaker encoder model
        device: Device to run inference on
        
    Returns:
        Speaker embeddings of shape (batch_size, embedding_size)
    """
    model.eval()
    with torch.no_grad():
        audio_batch = audio_batch.to(device)
        
        if isinstance(model, EncoderClassifier):
            # SpeechBrain model
            embeddings = model.encode_batch(audio_batch)
        else:
            # Custom ECAPA-TDNN model
            # First extract mel-spectrogram features
            mel_features = extract_mel_features(audio_batch)
            embeddings = model(mel_features)
    
    return embeddings

def extract_mel_features(audio_batch, sample_rate=16000, n_mels=80):
    """
    Extract mel-spectrogram features from audio
    
    Args:
        audio_batch: Audio tensor
        sample_rate: Audio sample rate
        n_mels: Number of mel bands
        
    Returns:
        Mel-spectrogram features
    """
    import librosa
    
    mel_features = []
    for audio in audio_batch:
        # Convert to numpy
        audio_np = audio.cpu().numpy()
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_np,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert to tensor
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float32)
        mel_features.append(mel_tensor)
    
    # Pad sequences to same length
    max_length = max(mel.shape[1] for mel in mel_features)
    padded_features = []
    
    for mel in mel_features:
        if mel.shape[1] < max_length:
            # Pad with zeros
            padding = torch.zeros(mel.shape[0], max_length - mel.shape[1])
            mel_padded = torch.cat([mel, padding], dim=1)
        else:
            mel_padded = mel
        padded_features.append(mel_padded)
    
    # Stack into batch
    batch_features = torch.stack(padded_features)
    
    return batch_features

# Utility functions for speaker similarity
def calculate_speaker_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two speaker embeddings
    
    Args:
        embedding1: First speaker embedding
        embedding2: Second speaker embedding
        
    Returns:
        Similarity score between 0 and 1
    """
    similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
    return similarity.item()

def verify_speaker_identity(embedding1, embedding2, threshold=0.7):
    """
    Verify if two embeddings belong to the same speaker
    
    Args:
        embedding1: First speaker embedding
        embedding2: Second speaker embedding
        threshold: Similarity threshold
        
    Returns:
        True if same speaker, False otherwise
    """
    similarity = calculate_speaker_similarity(embedding1, embedding2)
    return similarity >= threshold 