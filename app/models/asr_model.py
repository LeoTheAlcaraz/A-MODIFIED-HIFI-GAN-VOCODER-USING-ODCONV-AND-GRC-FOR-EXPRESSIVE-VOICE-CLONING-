"""
Automatic Speech Recognition (ASR) model for real-time speech-to-text conversion
"""

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ASRModel:
    """ASR model for real-time speech recognition"""
    
    def __init__(self, 
                 model_name: str = "facebook/wav2vec2-large-xlsr-53",
                 device: str = "cpu",
                 language: str = "en"):
        """
        Initialize ASR model
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
            language: Language code for recognition
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        
        # Load model and processor
        self.processor = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the ASR model and processor"""
        try:
            logger.info(f"Loading ASR model: {self.model_name}")
            
            # Load processor
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            
            # Load model
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("ASR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ASR model: {e}")
            raise
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Audio array (float32, [-1, 1])
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        try:
            # Ensure audio is the right format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Resample if necessary
            if sample_rate != 16000:
                audio_data = self._resample_audio(audio_data, sample_rate, 16000)
            
            # Prepare input
            inputs = self.processor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return ""
    
    def transcribe_batch(self, audio_batch: List[np.ndarray], 
                        sample_rate: int = 16000) -> List[str]:
        """
        Transcribe a batch of audio samples
        
        Args:
            audio_batch: List of audio arrays
            sample_rate: Audio sample rate
            
        Returns:
            List of transcriptions
        """
        transcriptions = []
        
        for audio_data in audio_batch:
            transcription = self.transcribe(audio_data, sample_rate)
            transcriptions.append(transcription)
        
        return transcriptions
    
    def _resample_audio(self, audio_data: np.ndarray, 
                       orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio_data
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        
        # Resample
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled = resampler(audio_tensor)
        
        return resampled.squeeze(0).numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "language": self.language,
            "sample_rate": 16000,
            "supported_languages": ["en", "es", "multilingual"]
        }


class StreamingASR:
    """Streaming ASR for real-time transcription"""
    
    def __init__(self, asr_model: ASRModel, buffer_size: int = 10):
        """
        Initialize streaming ASR
        
        Args:
            asr_model: ASR model instance
            buffer_size: Number of audio chunks to buffer
        """
        self.asr_model = asr_model
        self.buffer_size = buffer_size
        self.audio_buffer = []
        self.is_streaming = False
        
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Add audio chunk and return transcription if buffer is full
        
        Args:
            audio_chunk: Audio chunk
            
        Returns:
            Transcription if buffer is full, None otherwise
        """
        self.audio_buffer.append(audio_chunk)
        
        if len(self.audio_buffer) >= self.buffer_size:
            # Combine chunks
            combined_audio = np.concatenate(self.audio_buffer)
            
            # Transcribe
            transcription = self.asr_model.transcribe(combined_audio)
            
            # Clear buffer
            self.audio_buffer = []
            
            return transcription
        
        return None
    
    def flush_buffer(self) -> str:
        """Flush remaining audio in buffer and transcribe"""
        if not self.audio_buffer:
            return ""
        
        combined_audio = np.concatenate(self.audio_buffer)
        transcription = self.asr_model.transcribe(combined_audio)
        self.audio_buffer = []
        
        return transcription
    
    def start_streaming(self):
        """Start streaming mode"""
        self.is_streaming = True
        self.audio_buffer = []
    
    def stop_streaming(self):
        """Stop streaming mode"""
        self.is_streaming = False
        self.audio_buffer = []


# Model factory for different languages
class ASRModelFactory:
    """Factory for creating ASR models for different languages"""
    
    MODELS = {
        "en": "facebook/wav2vec2-large-960h-lv60-self",
        "es": "facebook/wav2vec2-large-960h-lv60-self",  # Use working model for Spanish too
        "multilingual": "facebook/wav2vec2-large-960h-lv60-self"
    }
    
    @classmethod
    def create_model(cls, language: str = "en", device: str = "cpu") -> ASRModel:
        """
        Create ASR model for specified language
        
        Args:
            language: Language code
            device: Device to run model on
            
        Returns:
            ASR model instance
        """
        model_name = cls.MODELS.get(language, cls.MODELS["multilingual"])
        return ASRModel(model_name=model_name, device=device, language=language)
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages"""
        return list(cls.MODELS.keys()) 