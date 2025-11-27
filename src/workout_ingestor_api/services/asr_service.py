"""Automatic Speech Recognition (ASR) service for video transcripts."""
import os
import subprocess
import tempfile
from typing import List, Dict, Optional
from fastapi import HTTPException

# Optional dependencies
try:
    import whisper  # type: ignore
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

try:
    from whisperx import load_model, load_align_model, align  # type: ignore
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False


class ASRService:
    """Service for transcribing audio using ASR."""
    
    @staticmethod
    def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional output path for audio file
            
        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav", prefix="audio_")
        
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", video_path,
                "-acodec", "pcm_s16le",
                "-ar", "16000",  # Whisper prefers 16kHz
                "-ac", "1",  # Mono
                output_path
            ],
            check=True
        )
        
        return output_path
    
    @staticmethod
    def transcribe_with_whisper(
        audio_path: str,
        model_size: str = "small",
        language: str = "en",
        word_timestamps: bool = False
    ) -> Dict:
        """
        Transcribe audio using OpenAI Whisper.
        
        Args:
            audio_path: Path to audio file
            model_size: Whisper model size (tiny, base, small, medium, large, large-v3)
            language: Language code (e.g., "en")
            word_timestamps: Whether to return word-level timestamps
            
        Returns:
            Dict with transcript segments and metadata
        """
        if not WHISPER_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="Whisper is not installed. Run: pip install openai-whisper"
            )
        
        try:
            model = whisper.load_model(model_size)
            result = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=word_timestamps
            )
            
            return {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language", language),
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Whisper transcription failed: {e}"
            )
    
    @staticmethod
    def transcribe_with_whisperx(
        audio_path: str,
        model_size: str = "small",
        language: str = "en",
        device: str = "cpu"
    ) -> Dict:
        """
        Transcribe audio using WhisperX (better alignment & diarization).
        
        Args:
            audio_path: Path to audio file
            model_size: Whisper model size
            language: Language code
            device: Device to use (cpu, cuda)
            
        Returns:
            Dict with transcript segments, word timestamps, and speaker diarization
        """
        if not WHISPERX_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="WhisperX is not installed. Run: pip install whisperx"
            )
        
        try:
            # Load model
            model = load_model(model_size, device=device, language=language)
            
            # Transcribe
            result = model.transcribe(audio_path)
            
            # Align for word-level timestamps
            model_a, metadata = load_align_model(language_code=language, device=device)
            result = align(result["segments"], model_a, metadata, audio_path, device=device)
            
            return {
                "text": " ".join([seg["text"] for seg in result["segments"]]),
                "segments": result.get("segments", []),
                "word_segments": result.get("word_segments", []),
                "language": language,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"WhisperX transcription failed: {e}"
            )
    
    @staticmethod
    def transcribe_with_api(
        audio_path: str,
        provider: str = "deepgram",
        api_key: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio using cloud ASR API.
        
        Supported providers: deepgram, assemblyai, google, aws
        
        Args:
            audio_path: Path to audio file
            provider: ASR provider name
            api_key: API key for the provider
            
        Returns:
            Dict with transcript segments
        """
        # This is a placeholder - full implementation would integrate with APIs
        raise HTTPException(
            status_code=501,
            detail=f"ASR API integration for {provider} not yet implemented. Use Whisper/WhisperX for now."
        )

