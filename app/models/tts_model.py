"""
Text-to-Speech (TTS) model for real-time speech synthesis
"""

import torch
import torchaudio
from transformers import AutoProcessor, AutoModel
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging
import soundfile as sf
import io
import wave

logger = logging.getLogger(__name__)


class TTSModel:
    """TTS model for real-time speech synthesis"""
    
    def __init__(self, 
                 model_name: str = "microsoft/speecht5_tts",
                 device: str = "cpu",
                 sample_rate: int = 22050):
        """
        Initialize TTS model
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
            sample_rate: Output audio sample rate
        """
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        
        # Load model and processor
        self.processor = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the TTS model and processor"""
        try:
            logger.info(f"Loading TTS model: {self.model_name}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading TTS model: {e}")
            raise
    
    def synthesize(self, text: str, speaker_id: Optional[int] = None) -> np.ndarray:
        """
        Synthesize speech from text
        
        Args:
            text: Input text to synthesize
            speaker_id: Speaker ID for voice cloning (if supported)
            
        Returns:
            Audio array (float32, [-1, 1])
        """
        try:
            if not text.strip():
                return np.array([])
            
            # Prepare input
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Add speaker ID if provided
            if speaker_id is not None:
                inputs["speaker_ids"] = torch.tensor([speaker_id]).to(self.device)
            
            # Generate speech
            with torch.no_grad():
                output = self.model.generate_speech(**inputs)
                speech = output.squeeze().cpu().numpy()
            
            # Normalize audio
            if np.max(np.abs(speech)) > 0:
                speech = speech / np.max(np.abs(speech))
            
            return speech.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            return np.array([])
    
    def synthesize_batch(self, texts: List[str], 
                        speaker_ids: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Synthesize speech for a batch of texts
        
        Args:
            texts: List of texts to synthesize
            speaker_ids: List of speaker IDs (optional)
            
        Returns:
            List of audio arrays
        """
        audio_samples = []
        
        for i, text in enumerate(texts):
            speaker_id = speaker_ids[i] if speaker_ids else None
            audio = self.synthesize(text, speaker_id)
            audio_samples.append(audio)
        
        return audio_samples
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "supported_languages": ["en", "es"]
        }


class MultilingualTTS:
    """Multilingual TTS with language-specific models"""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize multilingual TTS
        
        Args:
            device: Device to run models on
        """
        self.device = device
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load TTS models for different languages"""
        model_configs = {
            "en": "facebook/fastspeech2-en-ljspeech",
            "es": "facebook/fastspeech2-es-css10"
        }
        
        for lang, model_name in model_configs.items():
            try:
                self.models[lang] = TTSModel(
                    model_name=model_name,
                    device=self.device
                )
                logger.info(f"Loaded TTS model for {lang}")
            except Exception as e:
                logger.error(f"Failed to load TTS model for {lang}: {e}")
    
    def synthesize(self, text: str, language: str = "en", 
                  speaker_id: Optional[int] = None) -> np.ndarray:
        """
        Synthesize speech in specified language
        
        Args:
            text: Input text
            language: Language code
            speaker_id: Speaker ID for voice cloning
            
        Returns:
            Audio array
        """
        if language in self.models:
            return self.models[language].synthesize(text, speaker_id)
        else:
            logger.error(f"No TTS model available for {language}")
            return np.array([])
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        return list(self.models.keys())


class StreamingTTS:
    """Streaming TTS for real-time synthesis"""
    
    def __init__(self, tts_model: TTSModel, buffer_size: int = 3):
        """
        Initialize streaming TTS
        
        Args:
            tts_model: TTS model instance
            buffer_size: Number of text chunks to buffer
        """
        self.tts_model = tts_model
        self.buffer_size = buffer_size
        self.text_buffer = []
        self.is_streaming = False
        
    def add_text_chunk(self, text_chunk: str, 
                      speaker_id: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Add text chunk and return audio if buffer is full
        
        Args:
            text_chunk: Text chunk to synthesize
            speaker_id: Speaker ID for voice cloning
            
        Returns:
            Audio array if buffer is full, None otherwise
        """
        if text_chunk.strip():
            self.text_buffer.append(text_chunk)
        
        if len(self.text_buffer) >= self.buffer_size:
            # Combine text chunks
            combined_text = " ".join(self.text_buffer)
            
            # Synthesize speech
            audio = self.tts_model.synthesize(combined_text, speaker_id)
            
            # Clear buffer
            self.text_buffer = []
            
            return audio
        
        return None
    
    def flush_buffer(self, speaker_id: Optional[int] = None) -> np.ndarray:
        """Flush remaining text in buffer and synthesize"""
        if not self.text_buffer:
            return np.array([])
        
        combined_text = " ".join(self.text_buffer)
        audio = self.tts_model.synthesize(combined_text, speaker_id)
        self.text_buffer = []
        
        return audio
    
    def start_streaming(self):
        """Start streaming mode"""
        self.is_streaming = True
        self.text_buffer = []
    
    def stop_streaming(self):
        """Stop streaming mode"""
        self.is_streaming = False
        self.text_buffer = []


class AudioPostProcessor:
    """Audio post-processing utilities"""
    
    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        
        # Resample
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled = resampler(audio_tensor)
        
        return resampled.squeeze(0).numpy()
    
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    @staticmethod
    def trim_silence(audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Trim silence from beginning and end of audio"""
        import librosa
        return librosa.effects.trim(audio, top_db=top_db)[0]
    
    @staticmethod
    def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 22050) -> bytes:
        """Convert audio array to WAV bytes"""
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Create WAV file in memory
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            return wav_buffer.getvalue()


# Model factory for different languages
class TTSModelFactory:
    """Factory for creating TTS models for different languages"""
    
    MODELS = {
        "en": "microsoft/speecht5_tts",
        "es": "microsoft/speecht5_tts",
        "fr": "microsoft/speecht5_tts"
    }
    
    @classmethod
    def create_model(cls, language: str = "en", device: str = "cpu") -> TTSModel:
        """
        Create TTS model for specified language
        
        Args:
            language: Language code
            device: Device to run model on
            
        Returns:
            TTS model instance
        """
        model_name = cls.MODELS.get(language)
        
        if not model_name:
            raise ValueError(f"No TTS model available for {language}")
        
        return TTSModel(model_name=model_name, device=device)
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages"""
        return list(cls.MODELS.keys()) 