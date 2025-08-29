"""
Text translation model for real-time English-Spanish translation
"""

import torch
from transformers import MarianMTModel, MarianTokenizer, pipeline
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class TranslationModel:
    """Translation model for real-time text translation"""
    
    def __init__(self, 
                 model_name: str = "Helsinki-NLP/opus-mt-en-es",
                 device: str = "cpu",
                 max_length: int = 512):
        """
        Initialize translation model
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the translation model and tokenizer"""
        try:
            logger.info(f"Loading translation model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = MarianMTModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Translation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading translation model: {e}")
            raise
    
    def translate(self, text: str, source_lang: str = "en", target_lang: str = "es") -> str:
        """
        Translate text from source language to target language
        
        Args:
            text: Input text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        try:
            if not text.strip():
                return ""
            
            # Prepare input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode output
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translation.strip()
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return text  # Return original text if translation fails
    
    def translate_batch(self, texts: List[str], 
                       source_lang: str = "en", 
                       target_lang: str = "es") -> List[str]:
        """
        Translate a batch of texts
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        translations = []
        
        for text in texts:
            translation = self.translate(text, source_lang, target_lang)
            translations.append(translation)
        
        return translations
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Get supported language pairs"""
        return {
            "en": ["es"],  # English to Spanish
            "es": ["en"]   # Spanish to English
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "supported_languages": self.get_supported_languages()
        }


class TranslationPipeline:
    """Translation pipeline with multiple language support"""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize translation pipeline
        
        Args:
            device: Device to run models on
        """
        self.device = device
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load translation models for different language pairs"""
        model_configs = {
            "en-es": "Helsinki-NLP/opus-mt-en-es",
            "es-en": "Helsinki-NLP/opus-mt-es-en"
        }
        
        for lang_pair, model_name in model_configs.items():
            try:
                self.models[lang_pair] = TranslationModel(
                    model_name=model_name,
                    device=self.device
                )
                logger.info(f"Loaded translation model for {lang_pair}")
            except Exception as e:
                logger.error(f"Failed to load model for {lang_pair}: {e}")
    
    def translate(self, text: str, source_lang: str = "en", target_lang: str = "es") -> str:
        """
        Translate text using appropriate model
        
        Args:
            text: Input text
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Translated text
        """
        lang_pair = f"{source_lang}-{target_lang}"
        
        if lang_pair in self.models:
            return self.models[lang_pair].translate(text, source_lang, target_lang)
        else:
            logger.error(f"No translation model available for {lang_pair}")
            return text
    
    def get_available_language_pairs(self) -> List[str]:
        """Get available language pairs"""
        return list(self.models.keys())


class StreamingTranslator:
    """Streaming translator for real-time translation"""
    
    def __init__(self, translation_model: TranslationModel, buffer_size: int = 5):
        """
        Initialize streaming translator
        
        Args:
            translation_model: Translation model instance
            buffer_size: Number of text chunks to buffer
        """
        self.translation_model = translation_model
        self.buffer_size = buffer_size
        self.text_buffer = []
        self.is_streaming = False
        
    def add_text_chunk(self, text_chunk: str, 
                      source_lang: str = "en", 
                      target_lang: str = "es") -> Optional[str]:
        """
        Add text chunk and return translation if buffer is full
        
        Args:
            text_chunk: Text chunk to translate
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Translation if buffer is full, None otherwise
        """
        if text_chunk.strip():
            self.text_buffer.append(text_chunk)
        
        if len(self.text_buffer) >= self.buffer_size:
            # Combine text chunks
            combined_text = " ".join(self.text_buffer)
            
            # Translate
            translation = self.translation_model.translate(
                combined_text, source_lang, target_lang
            )
            
            # Clear buffer
            self.text_buffer = []
            
            return translation
        
        return None
    
    def flush_buffer(self, source_lang: str = "en", target_lang: str = "es") -> str:
        """Flush remaining text in buffer and translate"""
        if not self.text_buffer:
            return ""
        
        combined_text = " ".join(self.text_buffer)
        translation = self.translation_model.translate(
            combined_text, source_lang, target_lang
        )
        self.text_buffer = []
        
        return translation
    
    def start_streaming(self):
        """Start streaming mode"""
        self.is_streaming = True
        self.text_buffer = []
    
    def stop_streaming(self):
        """Stop streaming mode"""
        self.is_streaming = False
        self.text_buffer = []


# Model factory for different language pairs
class TranslationModelFactory:
    """Factory for creating translation models for different language pairs"""
    
    MODELS = {
        "en-es": "Helsinki-NLP/opus-mt-en-es",
        "es-en": "Helsinki-NLP/opus-mt-es-en",
        "en-fr": "Helsinki-NLP/opus-mt-en-fr",
        "fr-en": "Helsinki-NLP/opus-mt-fr-en"
    }
    
    @classmethod
    def create_model(cls, source_lang: str = "en", 
                    target_lang: str = "es", 
                    device: str = "cpu") -> TranslationModel:
        """
        Create translation model for specified language pair
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            device: Device to run model on
            
        Returns:
            Translation model instance
        """
        lang_pair = f"{source_lang}-{target_lang}"
        model_name = cls.MODELS.get(lang_pair)
        
        if not model_name:
            raise ValueError(f"No model available for {lang_pair}")
        
        return TranslationModel(model_name=model_name, device=device)
    
    @classmethod
    def get_supported_language_pairs(cls) -> List[str]:
        """Get list of supported language pairs"""
        return list(cls.MODELS.keys()) 