"""
Main FastAPI application for real-time voice translation
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional
import numpy as np
import base64
import io

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from .core.config import settings
from .core.translation_engine import RealTimeTranslationEngine, TranslationResult
from .core.audio_processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Real-time voice translation system"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Global translation engine
translation_engine: Optional[RealTimeTranslationEngine] = None
audio_processor: Optional[AudioProcessor] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        for client_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, client_id)

manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global translation_engine, audio_processor
    
    try:
        logger.info("Initializing translation engine...")
        
        # Initialize translation engine
        translation_engine = RealTimeTranslationEngine(
            device="cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu",
            source_lang=settings.default_source_lang,
            target_lang=settings.default_target_lang
        )
        
        # Initialize audio processor
        audio_processor = AudioProcessor(
            sample_rate=settings.sample_rate,
            chunk_size=settings.chunk_size
        )
        
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")


@app.get("/")
async def get_index():
    """Serve the main application page"""
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.version,
        "models_loaded": translation_engine is not None
    }


@app.get("/api/models/info")
async def get_model_info():
    """Get information about loaded models"""
    if not translation_engine:
        raise HTTPException(status_code=503, detail="Translation engine not initialized")
    
    return translation_engine.get_model_info()


@app.post("/api/translate/text")
async def translate_text(request: Dict[str, Any]):
    """Translate text endpoint"""
    if not translation_engine:
        raise HTTPException(status_code=503, detail="Translation engine not initialized")
    
    text = request.get("text", "")
    source_lang = request.get("source_lang", settings.default_source_lang)
    target_lang = request.get("target_lang", settings.default_target_lang)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        translated_text = translation_engine.translate_text(text)
        return {
            "source_text": text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang
        }
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        raise HTTPException(status_code=500, detail="Translation failed")


@app.post("/api/synthesize/text")
async def synthesize_text(request: Dict[str, Any]):
    """Synthesize text to speech endpoint"""
    if not translation_engine:
        raise HTTPException(status_code=503, detail="Translation engine not initialized")
    
    text = request.get("text", "")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        audio = translation_engine.synthesize_text(text)
        
        # Convert audio to base64
        audio_bytes = audio_processor.audio_to_wav_bytes(audio)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        return {
            "text": text,
            "audio": audio_b64,
            "sample_rate": settings.sample_rate
        }
    except Exception as e:
        logger.error(f"Error synthesizing text: {e}")
        raise HTTPException(status_code=500, detail="Synthesis failed")


@app.websocket("/ws/translate/{client_id}")
async def websocket_translate(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time translation"""
    await manager.connect(websocket, client_id)
    
    try:
        # Start streaming mode
        if translation_engine:
            translation_engine.start_streaming()
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "audio_chunk":
                # Process audio chunk
                await process_audio_chunk(message, client_id)
            
            elif message_type == "text_translate":
                # Process text translation
                await process_text_translation(message, client_id)
            
            elif message_type == "switch_languages":
                # Switch source and target languages
                await switch_languages(message, client_id)
            
            elif message_type == "ping":
                # Respond to ping
                await manager.send_personal_message({"type": "pong"}, client_id)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        if translation_engine:
            translation_engine.stop_streaming()
    
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)
        if translation_engine:
            translation_engine.stop_streaming()


async def process_audio_chunk(message: Dict[str, Any], client_id: str):
    """Process audio chunk from WebSocket"""
    try:
        # Decode base64 audio
        audio_b64 = message.get("audio", "")
        audio_bytes = base64.b64decode(audio_b64)
        
        # Convert to numpy array
        audio_array = audio_processor.wav_bytes_to_array(audio_bytes)
        
        # Process with translation engine
        if translation_engine:
            result = translation_engine.process_streaming_audio(audio_array)
            
            if result and result.translated_audio is not None:
                # Convert translated audio to base64
                translated_audio_bytes = audio_processor.audio_to_wav_bytes(result.translated_audio)
                translated_audio_b64 = base64.b64encode(translated_audio_bytes).decode("utf-8")
                
                # Send result to client
                response = {
                    "type": "translation_result",
                    "source_text": result.source_text,
                    "translated_text": result.translated_text,
                    "translated_audio": translated_audio_b64,
                    "source_language": result.source_language,
                    "target_language": result.target_language,
                    "processing_time": result.processing_time
                }
                
                await manager.send_personal_message(response, client_id)
    
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": "Error processing audio"
        }, client_id)


async def process_text_translation(message: Dict[str, Any], client_id: str):
    """Process text translation from WebSocket"""
    try:
        text = message.get("text", "")
        source_lang = message.get("source_lang", settings.default_source_lang)
        target_lang = message.get("target_lang", settings.default_target_lang)
        
        if translation_engine:
            # Translate text
            translated_text = translation_engine.translate_text(text)
            
            # Synthesize speech
            audio = translation_engine.synthesize_text(translated_text)
            
            # Convert to base64
            audio_bytes = audio_processor.audio_to_wav_bytes(audio)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            # Send result
            response = {
                "type": "text_translation_result",
                "source_text": text,
                "translated_text": translated_text,
                "translated_audio": audio_b64,
                "source_language": source_lang,
                "target_language": target_lang
            }
            
            await manager.send_personal_message(response, client_id)
    
    except Exception as e:
        logger.error(f"Error processing text translation: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": "Error translating text"
        }, client_id)


async def switch_languages(message: Dict[str, Any], client_id: str):
    """Switch source and target languages"""
    try:
        source_lang = message.get("source_lang", settings.default_source_lang)
        target_lang = message.get("target_lang", settings.default_target_lang)
        
        if translation_engine:
            translation_engine.switch_languages(source_lang, target_lang)
            
            await manager.send_personal_message({
                "type": "languages_switched",
                "source_language": source_lang,
                "target_language": target_lang
            }, client_id)
    
    except Exception as e:
        logger.error(f"Error switching languages: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": "Error switching languages"
        }, client_id)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    ) 