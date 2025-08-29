#!/usr/bin/env python3
"""
Evaluation Framework for Modified HiFi-GAN Vocoder
Implements the metrics from the thesis: SIM, AL, and ASR-BLEU
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Evaluation metrics for the modified HiFi-GAN vocoder"""
    
    def __init__(self, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        
    def compute_cosine_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding [B, D]
            embedding2: Second embedding [B, D]
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize embeddings
        embedding1_norm = F.normalize(embedding1, p=2, dim=1)
        embedding2_norm = F.normalize(embedding2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(embedding1_norm, embedding2_norm, dim=1)
        
        return similarity.mean().item()
    
    def compute_speaker_similarity(self, source_audio: torch.Tensor, 
                                 generated_audio: torch.Tensor,
                                 speaker_extractor) -> float:
        """
        Compute speaker similarity using ECAPA-TDNN embeddings
        
        Args:
            source_audio: Source audio waveform [B, 1, T]
            generated_audio: Generated audio waveform [B, 1, T]
            speaker_extractor: ECAPA-TDNN speaker extractor
            
        Returns:
            Speaker similarity score (0-1)
        """
        # Extract speaker embeddings
        with torch.no_grad():
            source_embedding, _ = speaker_extractor(source_audio)
            generated_embedding, _ = speaker_extractor(generated_audio)
        
        # Compute cosine similarity
        similarity = self.compute_cosine_similarity(source_embedding, generated_embedding)
        
        return similarity
    
    def compute_emotion_similarity(self, source_audio: torch.Tensor,
                                 generated_audio: torch.Tensor,
                                 emotion_extractor) -> float:
        """
        Compute emotion similarity using Emotion2Vec embeddings
        
        Args:
            source_audio: Source audio waveform [B, 1, T]
            generated_audio: Generated audio waveform [B, 1, T]
            emotion_extractor: Emotion2Vec emotion extractor
            
        Returns:
            Emotion similarity score (0-1)
        """
        # Extract emotion embeddings
        with torch.no_grad():
            _, source_emotion = emotion_extractor(source_audio)
            _, generated_emotion = emotion_extractor(generated_audio)
        
        # Compute cosine similarity
        similarity = self.compute_cosine_similarity(source_emotion, generated_emotion)
        
        return similarity
    
    def compute_average_lagging(self, source_timestamps: List[float],
                               target_timestamps: List[float]) -> float:
        """
        Compute Average Lagging (AL) metric for simultaneous translation
        
        Args:
            source_timestamps: Timestamps of source speech tokens
            target_timestamps: Timestamps of target speech generation
            
        Returns:
            Average lagging in milliseconds
        """
        if len(source_timestamps) != len(target_timestamps):
            raise ValueError("Source and target timestamps must have the same length")
        
        # Compute lagging for each token
        laggings = []
        for i, (src_time, tgt_time) in enumerate(zip(source_timestamps, target_timestamps)):
            lagging = tgt_time - src_time
            laggings.append(lagging)
        
        # Compute average lagging
        avg_lagging = np.mean(laggings)
        
        return avg_lagging
    
    def compute_asr_bleu(self, reference_text: str, generated_audio: torch.Tensor,
                         asr_model, tokenizer, bleu_scorer) -> float:
        """
        Compute ASR-BLEU score for translation quality
        
        Args:
            reference_text: Reference translation text
            generated_audio: Generated audio waveform [B, 1, T]
            asr_model: ASR model for transcription
            tokenizer: Tokenizer for text processing
            bleu_scorer: BLEU score calculator
            
        Returns:
            ASR-BLEU score (0-100)
        """
        # Transcribe generated audio
        with torch.no_grad():
            transcription = asr_model(generated_audio)
        
        # Tokenize reference and hypothesis
        reference_tokens = tokenizer(reference_text)
        hypothesis_tokens = tokenizer(transcription)
        
        # Compute BLEU score
        bleu_score = bleu_scorer(reference_tokens, [hypothesis_tokens])
        
        return bleu_score.score

class StreamSpeechEvaluator:
    """Evaluator for StreamSpeech with modified HiFi-GAN vocoder"""
    
    def __init__(self, system, embedding_extractors, asr_model, tokenizer, bleu_scorer):
        self.system = system
        self.embedding_extractors = embedding_extractors
        self.asr_model = asr_model
        self.tokenizer = tokenizer
        self.bleu_scorer = bleu_scorer
        self.metrics = EvaluationMetrics()
        
    def evaluate_single_sample(self, source_audio: torch.Tensor, 
                             reference_text: str,
                             speaker_embedding: Optional[torch.Tensor] = None,
                             emotion_embedding: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate a single audio sample
        
        Args:
            source_audio: Source audio waveform [B, 1, T]
            reference_text: Reference translation text
            speaker_embedding: Speaker embedding [B, 192]
            emotion_embedding: Emotion embedding [B, 256]
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Generate translation
        with torch.no_grad():
            outputs = self.system(source_audio, speaker_embedding=speaker_embedding, 
                                emotion_embedding=emotion_embedding)
        
        generated_audio = outputs['generated_waveform']
        
        # Extract embeddings for similarity computation
        if speaker_embedding is None:
            speaker_embedding = outputs['speaker_embedding']
        if emotion_embedding is None:
            emotion_embedding = outputs['emotion_embedding']
        
        # Compute metrics
        speaker_sim = self.metrics.compute_speaker_similarity(
            source_audio, generated_audio, self.embedding_extractors['speaker']
        )
        
        emotion_sim = self.metrics.compute_emotion_similarity(
            source_audio, generated_audio, self.embedding_extractors['emotion']
        )
        
        # Compute ASR-BLEU
        asr_bleu = self.metrics.compute_asr_bleu(
            reference_text, generated_audio, 
            self.asr_model, self.tokenizer, self.bleu_scorer
        )
        
        return {
            'speaker_similarity': speaker_sim,
            'emotion_similarity': emotion_sim,
            'asr_bleu': asr_bleu
        }
    
    def evaluate_batch(self, source_audios: torch.Tensor,
                      reference_texts: List[str],
                      speaker_embeddings: Optional[torch.Tensor] = None,
                      emotion_embeddings: Optional[torch.Tensor] = None) -> Dict[str, List[float]]:
        """
        Evaluate a batch of audio samples
        
        Args:
            source_audios: Source audio waveforms [B, 1, T]
            reference_texts: List of reference translation texts
            speaker_embeddings: Speaker embeddings [B, 192]
            emotion_embeddings: Emotion embeddings [B, 256]
            
        Returns:
            Dictionary containing evaluation metrics for each sample
        """
        batch_size = source_audios.size(0)
        results = {
            'speaker_similarity': [],
            'emotion_similarity': [],
            'asr_bleu': []
        }
        
        for i in range(batch_size):
            # Extract single sample
            source_audio = source_audios[i:i+1]
            reference_text = reference_texts[i]
            
            speaker_emb = speaker_embeddings[i:i+1] if speaker_embeddings is not None else None
            emotion_emb = emotion_embeddings[i:i+1] if emotion_embeddings is not None else None
            
            # Evaluate single sample
            sample_results = self.evaluate_single_sample(
                source_audio, reference_text, speaker_emb, emotion_emb
            )
            
            # Store results
            for metric, value in sample_results.items():
                results[metric].append(value)
        
        return results
    
    def compute_statistics(self, results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical summary of evaluation results
        
        Args:
            results: Dictionary containing metric lists
            
        Returns:
            Dictionary containing mean, std, min, max for each metric
        """
        statistics = {}
        
        for metric, values in results.items():
            values_array = np.array(values)
            statistics[metric] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array))
            }
        
        return statistics

class RealTimeEvaluator:
    """Real-time evaluation for streaming scenarios"""
    
    def __init__(self, system, chunk_size: int = 32):
        self.system = system
        self.chunk_size = chunk_size
        self.source_buffer = []
        self.target_buffer = []
        self.timestamps = []
        
    def process_chunk(self, audio_chunk: torch.Tensor, 
                     speaker_embedding: Optional[torch.Tensor] = None,
                     emotion_embedding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process a single audio chunk in real-time
        
        Args:
            audio_chunk: Audio chunk [B, T, F]
            speaker_embedding: Speaker embedding [B, 192]
            emotion_embedding: Emotion embedding [B, 256]
            
        Returns:
            Dictionary containing streaming outputs
        """
        start_time = time.time()
        
        # Process chunk through system
        outputs = self.system.streaming_forward(
            audio_chunk, speaker_embedding, emotion_embedding
        )
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Store timestamps for lagging computation
        self.timestamps.append({
            'source_time': start_time,
            'target_time': end_time,
            'processing_time': processing_time
        })
        
        return outputs
    
    def compute_streaming_metrics(self) -> Dict[str, float]:
        """
        Compute real-time metrics for streaming evaluation
        
        Returns:
            Dictionary containing streaming metrics
        """
        if len(self.timestamps) < 2:
            return {}
        
        # Compute average processing time
        processing_times = [t['processing_time'] for t in self.timestamps]
        avg_processing_time = np.mean(processing_times)
        
        # Compute average lagging
        source_times = [t['source_time'] for t in self.timestamps]
        target_times = [t['target_time'] for t in self.timestamps]
        
        avg_lagging = self.metrics.compute_average_lagging(source_times, target_times)
        
        return {
            'avg_processing_time_ms': avg_processing_time,
            'avg_lagging_ms': avg_lagging,
            'total_chunks': len(self.timestamps)
        }

def create_evaluation_report(results: Dict[str, List[float]], 
                           statistics: Dict[str, Dict[str, float]],
                           output_path: str):
    """
    Create and save evaluation report
    
    Args:
        results: Raw evaluation results
        statistics: Statistical summary
        output_path: Path to save report
    """
    report = {
        'evaluation_summary': {
            'total_samples': len(next(iter(results.values()))),
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics_evaluated': list(results.keys())
        },
        'raw_results': results,
        'statistics': statistics,
        'performance_analysis': {
            'speaker_similarity': {
                'benchmark_score': 0.73,  # From Wang et al. (2023)
                'threshold': 0.70,  # Acceptable threshold
                'achieved': statistics['speaker_similarity']['mean'],
                'status': 'PASS' if statistics['speaker_similarity']['mean'] >= 0.70 else 'FAIL'
            },
            'emotion_similarity': {
                'threshold': 0.70,
                'achieved': statistics['emotion_similarity']['mean'],
                'status': 'PASS' if statistics['emotion_similarity']['mean'] >= 0.70 else 'FAIL'
            },
            'asr_bleu': {
                'benchmark_score': 27.25,  # From Zhang et al. (2024)
                'threshold': 20.0,  # Basic fluency threshold
                'achieved': statistics['asr_bleu']['mean'],
                'status': 'PASS' if statistics['asr_bleu']['mean'] >= 20.0 else 'FAIL'
            }
        }
    }
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation report saved to {output_path}")

if __name__ == "__main__":
    # Test the evaluation framework
    print("Testing Evaluation Framework...")
    
    # Create dummy data for testing
    batch_size = 4
    seq_len = 100
    embedding_dim = 192
    
    source_audio = torch.randn(batch_size, 1, seq_len)
    generated_audio = torch.randn(batch_size, 1, seq_len)
    embedding1 = torch.randn(batch_size, embedding_dim)
    embedding2 = torch.randn(batch_size, embedding_dim)
    
    # Test cosine similarity
    metrics = EvaluationMetrics()
    similarity = metrics.compute_cosine_similarity(embedding1, embedding2)
    print(f"Cosine similarity: {similarity:.4f}")
    
    # Test average lagging
    source_times = [0.0, 1.0, 2.0, 3.0]
    target_times = [0.5, 1.5, 2.5, 3.5]
    avg_lagging = metrics.compute_average_lagging(source_times, target_times)
    print(f"Average lagging: {avg_lagging:.2f} seconds")
    
    print("✅ Evaluation Framework test successful!")
