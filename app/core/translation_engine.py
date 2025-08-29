"""
Main translation engine for real-time voice translation
"""

import asyncio
import numpy as np
from typing import Optional, Dict, Any, Callable, List
import logging
from dataclasses import dataclass
from enum import Enum
import time

from ..models.asr_model import ASRModel, StreamingASR, ASRModelFactory
from ..models.translation_model import TranslationModel, StreamingTranslator
from ..models.tts_model import TTSModel, StreamingTTS, AudioPostProcessor
from ..core.audio_processor import AudioProcessor, RealTimeAudioStream

logger = logging.getLogger(__name__)


class TranslationMode(Enum):
    """Translation modes"""
    REALTIME = "realtime"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class TranslationResult:
    """Result of translation process"""
    source_text: str
    translated_text: str
    source_audio: Optional[np.ndarray] = None
    translated_audio: Optional[np.ndarray] = None
    source_language: str = "en"
    target_language: str = "es"
    processing_time: float = 0.0
    confidence: float = 0.0


class RealTimeTranslationEngine:
    """Real-time voice translation engine"""
    
    def __init__(self, 
                 device: str = "cpu",
                 source_lang: str = "en",
                 target_lang: str = "es",
                 mode: TranslationMode = TranslationMode.REALTIME):
        """
        Initialize translation engine
        
        Args:
            device: Device to run models on
            source_lang: Source language
            target_lang: Target language
            mode: Translation mode
        """
        self.device = device
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.mode = mode
        
        # Initialize models
        self.asr_model = None
        self.translation_model = None
        self.tts_model = None
        self.audio_processor = None
        
        # Streaming components
        self.streaming_asr = None
        self.streaming_translator = None
        self.streaming_tts = None
        
        # Callbacks
        self.on_transcription = None
        self.on_translation = None
        self.on_synthesis = None
        self.on_result = None
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load all required models"""
        try:
            logger.info("Loading translation engine models...")
            
            # Load ASR model appropriate for the source language
            self.asr_model = ASRModelFactory.create_model(
                language=self.source_lang,
                device=self.device
            )
            
            # Load translation model
            self.translation_model = TranslationModel(
                model_name="Helsinki-NLP/opus-mt-en-es" if self.source_lang == "en" else "Helsinki-NLP/opus-mt-es-en",
                device=self.device
            )
            
            # Load TTS model (using working models)
            self.tts_model = TTSModel(
                model_name="microsoft/speecht5_tts",  # Works for both languages
                device=self.device
            )
            
            # Initialize audio processor
            self.audio_processor = AudioProcessor(
                sample_rate=16000,
                chunk_size=1024
            )
            
            # Initialize streaming components
            self.streaming_asr = StreamingASR(self.asr_model, buffer_size=5)
            self.streaming_translator = StreamingTranslator(self.translation_model, buffer_size=3)
            self.streaming_tts = StreamingTTS(self.tts_model, buffer_size=2)
            
            logger.info("Translation engine models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading translation engine models: {e}")
            raise
    
    def translate_audio(self, audio_data: np.ndarray, 
                       sample_rate: int = 16000) -> TranslationResult:
        """
        Translate audio from source language to target language
        
        Args:
            audio_data: Input audio array
            sample_rate: Audio sample rate
            
        Returns:
            Translation result
        """
        start_time = time.time()
        
        try:
            # Preprocess audio
            processed_audio = self.audio_processor.preprocess_audio(audio_data)
            
            # Step 1: Speech recognition
            source_text = self.asr_model.transcribe(processed_audio, sample_rate)
            
            if self.on_transcription:
                self.on_transcription(source_text)
            
            if not source_text.strip():
                return TranslationResult(
                    source_text="",
                    translated_text="",
                    source_audio=audio_data,
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Text translation
            translated_text = self.translation_model.translate(
                source_text, self.source_lang, self.target_lang
            )
            
            if self.on_translation:
                self.on_translation(translated_text)
            
            # Step 3: Speech synthesis
            translated_audio = self.tts_model.synthesize(translated_text)
            
            if self.on_synthesis:
                self.on_synthesis(translated_audio)
            
            # Post-process audio
            translated_audio = AudioPostProcessor.normalize_audio(translated_audio)
            translated_audio = AudioPostProcessor.trim_silence(translated_audio)
            
            result = TranslationResult(
                source_text=source_text,
                translated_text=translated_text,
                source_audio=audio_data,
                translated_audio=translated_audio,
                source_language=self.source_lang,
                target_language=self.target_lang,
                processing_time=time.time() - start_time
            )
            
            if self.on_result:
                self.on_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return TranslationResult(
                source_text="",
                translated_text="",
                source_audio=audio_data,
                processing_time=time.time() - start_time
            )
    
    def translate_text(self, text: str, source_lang: str = None, target_lang: str = None) -> TranslationResult:
        """
        Translate text from source to target language
        
        Args:
            text: Input text
            source_lang: Source language (optional, uses instance default)
            target_lang: Target language (optional, uses instance default)
            
        Returns:
            TranslationResult with translated text
        """
        if not self.translation_model:
            raise RuntimeError("Translation model not loaded")
        
        start_time = time.time()
        
        # Use provided languages or defaults
        src_lang = source_lang or self.source_lang
        tgt_lang = target_lang or self.target_lang
        
        translated_text = self.translation_model.translate(text, src_lang, tgt_lang)
        
        processing_time = time.time() - start_time
        
        return TranslationResult(
            source_text=text,
            translated_text=translated_text,
            source_language=src_lang,
            target_language=tgt_lang,
            processing_time=processing_time,
            confidence=0.9  # Placeholder confidence
        )
    
    def synthesize_text(self, text: str) -> np.ndarray:
        """
        Synthesize text to speech
        
        Args:
            text: Input text
            
        Returns:
            Audio array
        """
        return self.tts_model.synthesize(text)
    
    def start_streaming(self):
        """Start streaming translation mode"""
        self.streaming_asr.start_streaming()
        self.streaming_translator.start_streaming()
        self.streaming_tts.start_streaming()
        logger.info("Started streaming translation mode")
    
    def stop_streaming(self):
        """Stop streaming translation mode"""
        self.streaming_asr.stop_streaming()
        self.streaming_translator.stop_streaming()
        self.streaming_tts.stop_streaming()
        logger.info("Stopped streaming translation mode")
    
    def process_streaming_audio(self, audio_chunk: np.ndarray) -> Optional[TranslationResult]:
        """
        Process audio chunk in streaming mode
        
        Args:
            audio_chunk: Audio chunk
            
        Returns:
            Translation result if available, None otherwise
        """
        try:
            # Process with streaming ASR
            transcription = self.streaming_asr.add_audio_chunk(audio_chunk)
            
            if transcription:
                # Process with streaming translator
                translation = self.streaming_translator.add_text_chunk(
                    transcription, self.source_lang, self.target_lang
                )
                
                if translation:
                    # Process with streaming TTS
                    audio = self.streaming_tts.add_text_chunk(translation)
                    
                    if audio is not None:
                        return TranslationResult(
                            source_text=transcription,
                            translated_text=translation,
                            translated_audio=audio,
                            source_language=self.source_lang,
                            target_language=self.target_lang
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in streaming processing: {e}")
            return None
    
    def flush_streaming_buffers(self) -> TranslationResult:
        """Flush all streaming buffers and return final result"""
        try:
            # Flush ASR buffer
            final_transcription = self.streaming_asr.flush_buffer()
            
            if not final_transcription:
                return TranslationResult(
                    source_text="",
                    translated_text="",
                    source_language=self.source_lang,
                    target_language=self.target_lang
                )
            
            # Flush translation buffer
            final_translation = self.streaming_translator.flush_buffer(
                self.source_lang, self.target_lang
            )
            
            # Flush TTS buffer
            final_audio = self.streaming_tts.flush_buffer()
            
            return TranslationResult(
                source_text=final_transcription,
                translated_text=final_translation,
                translated_audio=final_audio,
                source_language=self.source_lang,
                target_language=self.target_lang
            )
            
        except Exception as e:
            logger.error(f"Error flushing streaming buffers: {e}")
            return TranslationResult(
                source_text="",
                translated_text="",
                source_language=self.source_lang,
                target_language=self.target_lang
            )
    
    def set_callbacks(self, 
                     on_transcription: Optional[Callable[[str], None]] = None,
                     on_translation: Optional[Callable[[str], None]] = None,
                     on_synthesis: Optional[Callable[[np.ndarray], None]] = None,
                     on_result: Optional[Callable[[TranslationResult], None]] = None):
        """Set callback functions for real-time processing"""
        self.on_transcription = on_transcription
        self.on_translation = on_translation
        self.on_synthesis = on_synthesis
        self.on_result = on_result
    
    def switch_languages(self, source_lang: str, target_lang: str):
        """Switch source and target languages"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Reload models if necessary
        self._load_models()
        
        logger.info(f"Switched languages: {source_lang} -> {target_lang}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "asr_model": self.asr_model.get_model_info() if self.asr_model else None,
            "translation_model": self.translation_model.get_model_info() if self.translation_model else None,
            "tts_model": self.tts_model.get_model_info() if self.tts_model else None,
            "source_language": self.source_lang,
            "target_language": self.target_lang,
            "mode": self.mode.value,
            "device": self.device
        }


class TranslationEngineFactory:
    """Factory for creating translation engines"""
    
    @staticmethod
    def create_engine(source_lang: str = "en", 
                     target_lang: str = "es", 
                     device: str = "cpu",
                     mode: TranslationMode = TranslationMode.REALTIME) -> RealTimeTranslationEngine:
        """
        Create translation engine with specified configuration
        
        Args:
            source_lang: Source language
            target_lang: Target language
            device: Device to run models on
            mode: Translation mode
            
        Returns:
            Translation engine instance
        """
        return RealTimeTranslationEngine(
            device=device,
            source_lang=source_lang,
            target_lang=target_lang,
            mode=mode
        )
    
    @staticmethod
    def create_multilingual_engine(device: str = "cpu") -> Dict[str, RealTimeTranslationEngine]:
        """Create engines for multiple language pairs"""
        language_pairs = [
            ("en", "es"),
            ("es", "en")
        ]
        
        engines = {}
        for source, target in language_pairs:
            engines[f"{source}-{target}"] = RealTimeTranslationEngine(
                device=device,
                source_lang=source,
                target_lang=target
            )
        
        return engines 