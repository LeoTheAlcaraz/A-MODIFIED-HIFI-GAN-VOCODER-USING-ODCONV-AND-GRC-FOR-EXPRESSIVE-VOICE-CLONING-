"""
Offline functionality manager for the voice translation system
"""

import os
import json
import logging
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoTokenizer, AutoModel
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OfflineManager:
    """Manages offline functionality and model caching"""
    
    def __init__(self, cache_dir: str = "models/cache", data_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.data_dir = Path(data_dir)
        self.models_dir = self.cache_dir / "models"
        self.datasets_dir = self.data_dir / "datasets"
        self.translations_dir = self.data_dir / "translations"
        
        # Create directories
        self._create_directories()
        
        # Model registry for offline models - Using our local trained models
        self.model_registry = {
            "asr": {
                "models/asr/en": {
                    "local_path": "models/asr/en",
                    "size_mb": 1200,
                    "languages": ["en"]
                },
                "models/asr/es": {
                    "local_path": "models/asr/es",
                    "size_mb": 1200,
                    "languages": ["es"]
                }
            },
            "translation": {
                "models/translation/en-es": {
                    "local_path": "models/translation/en-es",
                    "size_mb": 300,
                    "languages": ["en", "es"]
                },
                "models/translation/es-en": {
                    "local_path": "models/translation/es-en", 
                    "size_mb": 300,
                    "languages": ["es", "en"]
                }
            },
            "tts": {
                "models/tts/en": {
                    "local_path": "models/tts/en",
                    "size_mb": 500,
                    "languages": ["en"]
                },
                "models/tts/es": {
                    "local_path": "models/tts/es",
                    "size_mb": 500,
                    "languages": ["es"]
                }
            },
            "vocoder": {
                "models/hifigan_modified": {
                    "local_path": "models/hifigan_modified",
                    "size_mb": 200,
                    "languages": ["en", "es"]
                }
            }
        }
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.cache_dir,
            self.models_dir,
            self.datasets_dir,
            self.translations_dir,
            self.cache_dir / "asr",
            self.cache_dir / "translation", 
            self.cache_dir / "tts",
            self.cache_dir / "vocoder"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def is_model_available_offline(self, model_type: str, model_name: str) -> bool:
        """Check if a model is available offline"""
        if model_type not in self.model_registry:
            return False
        
        if model_name not in self.model_registry[model_type]:
            return False
        
        local_path = Path(self.model_registry[model_type][model_name]["local_path"])
        # Check if directory exists
        if not local_path.exists():
            return False
        
        # For vocoder models, check if source code is available
        if model_type == "vocoder":
            return any((local_path / file).exists() for file in ["__init__.py", "generator.py", "discriminators.py"])
        
        # For other models, check for common model files
        model_files = ["config.json", "model.safetensors", "pytorch_model.bin"]
        has_model_files = any((local_path / file).exists() for file in model_files)
        
        return has_model_files
    
    def get_offline_model_path(self, model_type: str, model_name: str) -> Optional[str]:
        """Get the local path for an offline model"""
        if self.is_model_available_offline(model_type, model_name):
            return str(Path(self.model_registry[model_type][model_name]["local_path"]))
        return None
    
    def download_model(self, model_type: str, model_name: str, force: bool = False) -> bool:
        """Download a model for offline use"""
        if model_type not in self.model_registry or model_name not in self.model_registry[model_type]:
            logger.error(f"Model {model_name} not found in registry for type {model_type}")
            return False
        
        model_info = self.model_registry[model_type][model_name]
        local_path = Path(model_info["local_path"])
        
        if local_path.exists() and not force:
            logger.info(f"Model {model_name} already exists locally")
            return True
        
        try:
            logger.info(f"Downloading {model_name} for offline use...")
            
            # Create directory
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Download model files
            if model_type == "asr":
                self._download_asr_model(model_name, local_path)
            elif model_type == "translation":
                self._download_translation_model(model_name, local_path)
            elif model_type == "tts":
                self._download_tts_model(model_name, local_path)
            elif model_type == "vocoder":
                self._download_vocoder_model(model_name, local_path)
            
            logger.info(f"Successfully downloaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False
    
    def _download_asr_model(self, model_name: str, local_path: Path):
        """Download ASR model"""
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        processor.save_pretrained(local_path)
        model.save_pretrained(local_path)
    
    def _download_translation_model(self, model_name: str, local_path: Path):
        """Download translation model"""
        from transformers import MarianMTModel, MarianTokenizer
        
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
    
    def _download_tts_model(self, model_name: str, local_path: Path):
        """Download TTS model"""
        from transformers import FastSpeech2ForTextToSpeech, FastSpeech2Processor
        
        processor = FastSpeech2Processor.from_pretrained(model_name)
        model = FastSpeech2ForTextToSpeech.from_pretrained(model_name)
        
        processor.save_pretrained(local_path)
        model.save_pretrained(local_path)
    
    def _download_vocoder_model(self, model_name: str, local_path: Path):
        """Download vocoder model - placeholder for custom HiFi-GAN"""
        # This would need to be implemented based on your custom HiFi-GAN
        logger.warning(f"Vocoder download not implemented for {model_name}")
    
    def download_all_models(self, force: bool = False) -> Dict[str, bool]:
        """Download all models for offline use"""
        results = {}
        
        for model_type, models in self.model_registry.items():
            for model_name in models:
                logger.info(f"Downloading {model_type}/{model_name}...")
                results[f"{model_type}/{model_name}"] = self.download_model(
                    model_type, model_name, force
                )
        
        return results
    
    def get_offline_translation_history(self) -> List[Dict[str, Any]]:
        """Get translation history from local storage"""
        history_file = self.translations_dir / "history.json"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_translation_to_history(self, translation: Dict[str, Any]):
        """Save translation to local history"""
        history = self.get_offline_translation_history()
        history.append({
            **translation,
            "timestamp": translation.get("timestamp", str(torch.tensor(0).item()))
        })
        
        # Keep only last 1000 translations
        if len(history) > 1000:
            history = history[-1000:]
        
        history_file = self.translations_dir / "history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def clear_translation_history(self):
        """Clear translation history"""
        history_file = self.translations_dir / "history.json"
        if history_file.exists():
            history_file.unlink()
    
    def get_cache_size(self) -> Dict[str, float]:
        """Get size of cached models in MB"""
        sizes = {}
        
        for model_type in ["asr", "translation", "tts", "vocoder"]:
            type_dir = self.cache_dir / model_type
            if type_dir.exists():
                total_size = sum(
                    f.stat().st_size for f in type_dir.rglob('*') if f.is_file()
                )
                sizes[model_type] = total_size / (1024 * 1024)  # Convert to MB
        
        return sizes
    
    def clear_cache(self, model_type: Optional[str] = None):
        """Clear model cache"""
        if model_type:
            cache_dir = self.cache_dir / model_type
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared cache for {model_type}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self._create_directories()
                logger.info("Cleared all cache")
    
    def check_offline_capability(self) -> Dict[str, Any]:
        """Check what's available offline"""
        capability = {
            "models": {},
            "total_cache_size_mb": 0,
            "can_work_offline": True
        }
        
        for model_type, models in self.model_registry.items():
            capability["models"][model_type] = {}
            for model_name in models:
                is_available = self.is_model_available_offline(model_type, model_name)
                capability["models"][model_type][model_name] = is_available
                if not is_available:
                    capability["can_work_offline"] = False
        
        # Calculate cache size
        cache_sizes = self.get_cache_size()
        capability["total_cache_size_mb"] = sum(cache_sizes.values())
        capability["cache_sizes"] = cache_sizes
        
        return capability


# Global offline manager instance
offline_manager = OfflineManager() 