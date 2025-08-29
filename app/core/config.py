"""
Configuration settings for the real-time voice translation system
"""

import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
import yaml


class AudioConfig:
    """Audio processing configuration"""
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    CHANNELS = 1
    FORMAT = "int16"
    VAD_MODE = 3  # Aggressiveness level (0-3)
    SILENCE_THRESHOLD = 0.1
    MIN_AUDIO_LENGTH = 0.5  # seconds
    MAX_AUDIO_LENGTH = 30.0  # seconds


class ModelConfig:
    """Model configuration"""
    # ASR Models - Using our local trained models
    ASR_MODEL_EN = "models/asr/en"
    ASR_MODEL_ES = "models/asr/es"
    ASR_DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
    # Translation Models - Using our local trained models
    TRANSLATION_MODEL_EN_TO_ES = "models/translation/en-es"
    TRANSLATION_MODEL_ES_TO_EN = "models/translation/es-en"
    TRANSLATION_DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
    # TTS Models - Using our local trained models
    TTS_MODEL_EN = "models/tts/en"
    TTS_MODEL_ES = "models/tts/es"
    TTS_DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
    # Vocoder Model - Using our modified HiFi-GAN
    VOCODER_MODEL_PATH = "models/hifigan_modified"
    VOCODER_DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"


class TranslationConfig:
    """Translation settings"""
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "es": "Spanish"
    }
    DEFAULT_SOURCE_LANG = "en"
    DEFAULT_TARGET_LANG = "es"
    MAX_TEXT_LENGTH = 512
    BATCH_SIZE = 1


class WebConfig:
    """Web application configuration"""
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    CORS_ORIGINS = ["*"]  # In production, specify actual origins


class Settings(BaseSettings):
    """Main application settings"""
    # Application
    app_name: str = "Real-Time Voice Translation"
    version: str = "1.0.0"
    debug: bool = False
    
    # Audio
    sample_rate: int = AudioConfig.SAMPLE_RATE
    chunk_size: int = AudioConfig.CHUNK_SIZE
    channels: int = AudioConfig.CHANNELS
    
    # Models
    asr_model_en: str = ModelConfig.ASR_MODEL_EN
    asr_model_es: str = ModelConfig.ASR_MODEL_ES
    translation_model_en_to_es: str = ModelConfig.TRANSLATION_MODEL_EN_TO_ES
    translation_model_es_to_en: str = ModelConfig.TRANSLATION_MODEL_ES_TO_EN
    tts_model_en: str = ModelConfig.TTS_MODEL_EN
    tts_model_es: str = ModelConfig.TTS_MODEL_ES
    vocoder_model_path: str = ModelConfig.VOCODER_MODEL_PATH
    
    # Translation
    supported_languages: Dict[str, str] = TranslationConfig.SUPPORTED_LANGUAGES
    default_source_lang: str = TranslationConfig.DEFAULT_SOURCE_LANG
    default_target_lang: str = TranslationConfig.DEFAULT_TARGET_LANG
    
    # Web
    host: str = WebConfig.HOST
    port: int = WebConfig.PORT
    cors_origins: list = WebConfig.CORS_ORIGINS
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def load_config(config_path: str = "configs/app_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


# Global settings instance
settings = Settings() 