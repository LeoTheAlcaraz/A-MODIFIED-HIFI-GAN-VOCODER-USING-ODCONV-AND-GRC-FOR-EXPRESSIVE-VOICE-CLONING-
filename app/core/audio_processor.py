"""
Audio processing utilities for real-time voice translation
"""

import numpy as np
import librosa
import soundfile as sf
import webrtcvad
import wave
import io
from typing import Generator, Optional, Tuple, List
from collections import deque
import threading
import time


class AudioProcessor:
    """Real-time audio processing with voice activity detection"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 channels: int = 1,
                 vad_mode: int = 3,
                 silence_threshold: float = 0.1,
                 min_audio_length: float = 0.5,
                 max_audio_length: float = 30.0):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Audio sample rate
            chunk_size: Size of audio chunks to process
            channels: Number of audio channels
            vad_mode: Voice activity detection aggressiveness (0-3)
            silence_threshold: Threshold for silence detection
            min_audio_length: Minimum audio length in seconds
            max_audio_length: Maximum audio length in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(vad_mode)
        
        # Audio buffer for real-time processing
        self.audio_buffer = deque(maxlen=int(max_audio_length * sample_rate))
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        
        # Threading for real-time processing
        self.processing_thread = None
        self.stop_processing = False
        
    def process_audio_chunk(self, audio_chunk: bytes) -> Optional[np.ndarray]:
        """
        Process a single audio chunk and detect voice activity
        
        Args:
            audio_chunk: Raw audio bytes
            
        Returns:
            Processed audio array if speech detected, None otherwise
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Add to buffer
        self.audio_buffer.extend(audio_array)
        
        # Check for voice activity
        is_speech = self._detect_speech(audio_chunk)
        
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            self.is_speaking = True
        else:
            self.silence_frames += 1
            
        # Check if we should return audio (speech ended or buffer full)
        should_return = self._should_return_audio()
        
        if should_return and len(self.audio_buffer) > 0:
            # Convert buffer to numpy array
            audio_data = np.array(list(self.audio_buffer), dtype=np.float32) / 32768.0
            
            # Clear buffer
            self.audio_buffer.clear()
            self.is_speaking = False
            self.speech_frames = 0
            self.silence_frames = 0
            
            return audio_data
            
        return None
    
    def _detect_speech(self, audio_chunk: bytes) -> bool:
        """Detect if audio chunk contains speech"""
        try:
            # VAD expects 10, 20, or 30ms frames
            frame_duration_ms = 30
            frame_size = int(self.sample_rate * frame_duration_ms / 1000)
            
            if len(audio_chunk) >= frame_size:
                return self.vad.is_speech(audio_chunk[:frame_size], self.sample_rate)
            return False
        except Exception:
            return False
    
    def detect_voice_activity(self, audio_chunk: bytes) -> bool:
        """Public method to detect voice activity (alias for _detect_speech)"""
        return self._detect_speech(audio_chunk)
    
    def _should_return_audio(self) -> bool:
        """Determine if we should return the buffered audio"""
        # Return if we have enough speech and silence
        min_speech_frames = int(self.min_audio_length * self.sample_rate / self.chunk_size)
        silence_threshold_frames = int(0.5 * self.sample_rate / self.chunk_size)  # 0.5s silence
        
        has_enough_speech = self.speech_frames >= min_speech_frames
        has_silence = self.silence_frames >= silence_threshold_frames
        
        return has_enough_speech and has_silence
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for model input
        
        Args:
            audio_data: Raw audio array
            
        Returns:
            Preprocessed audio array
        """
        # Ensure correct sample rate
        if len(audio_data) > 0:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=self.sample_rate, 
                target_sr=16000
            )
        
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Remove silence from beginning and end
        audio_data = librosa.effects.trim(audio_data, top_db=20)[0]
        
        return audio_data
    
    def audio_to_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """
        Convert audio array to WAV bytes
        
        Args:
            audio_data: Audio array
            
        Returns:
            WAV bytes
        """
        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            return wav_buffer.getvalue()
    
    def wav_bytes_to_array(self, wav_bytes: bytes) -> np.ndarray:
        """
        Convert WAV bytes to audio array
        
        Args:
            wav_bytes: WAV file bytes
            
        Returns:
            Audio array
        """
        with io.BytesIO(wav_bytes) as wav_buffer:
            with wave.open(wav_buffer, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                return audio_data.astype(np.float32) / 32768.0
    
    def resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio_data: Input audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio_data
        
        return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)


class RealTimeAudioStream:
    """Real-time audio streaming with buffering"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 buffer_size: int = 10):
        """
        Initialize real-time audio stream
        
        Args:
            sample_rate: Audio sample rate
            chunk_size: Size of audio chunks
            buffer_size: Number of chunks to buffer
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=buffer_size)
        self.is_streaming = False
        
    def add_chunk(self, audio_chunk: bytes) -> None:
        """Add audio chunk to buffer"""
        self.audio_buffer.append(audio_chunk)
    
    def get_buffered_audio(self) -> Optional[np.ndarray]:
        """Get all buffered audio as numpy array"""
        if not self.audio_buffer:
            return None
            
        # Combine all chunks
        combined_chunks = b''.join(self.audio_buffer)
        audio_data = np.frombuffer(combined_chunks, dtype=np.int16)
        
        # Clear buffer
        self.audio_buffer.clear()
        
        return audio_data.astype(np.float32) / 32768.0
    
    def start_streaming(self) -> None:
        """Start audio streaming"""
        self.is_streaming = True
    
    def stop_streaming(self) -> None:
        """Stop audio streaming"""
        self.is_streaming = False
        self.audio_buffer.clear()


def create_audio_chunks(audio_data: np.ndarray, 
                       chunk_size: int = 1024) -> Generator[np.ndarray, None, None]:
    """
    Split audio data into chunks
    
    Args:
        audio_data: Audio array
        chunk_size: Size of each chunk
        
    Yields:
        Audio chunks
    """
    for i in range(0, len(audio_data), chunk_size):
        yield audio_data[i:i + chunk_size]


def merge_audio_chunks(chunks: List[np.ndarray]) -> np.ndarray:
    """
    Merge audio chunks into single array
    
    Args:
        chunks: List of audio chunks
        
    Returns:
        Merged audio array
    """
    if not chunks:
        return np.array([])
    
    return np.concatenate(chunks) 