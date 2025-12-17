"""
Audio Transcription Module
Transcribes audio files to text using OpenAI Whisper
"""

import whisper
from pydub import AudioSegment
from typing import Dict, Tuple
import os


_whisper_model = None


def get_whisper_model():
    global _whisper_model
    
    if _whisper_model is None:
        print("Loading Whisper model (first time only, this may take a moment)...")
        _whisper_model = whisper.load_model("base")
        print("Whisper model loaded successfully!")
    
    return _whisper_model


def extract_text_from_audio(audio_path: str) -> Tuple[str, Dict]:
    try:
        print(f"Loading audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        
        duration_milliseconds = len(audio)
        duration_seconds = duration_milliseconds / 1000.0
        duration_minutes = duration_seconds / 60.0
        
        file_size_bytes = os.path.getsize(audio_path)
        file_size_kb = file_size_bytes / 1024.0
        
        print(f"Audio duration: {duration_minutes:.2f} minutes")
        print(f"File size: {file_size_kb:.2f} KB")
        
        model = get_whisper_model()
        
        print(f"Transcribing audio (this may take a few moments)...")
        result = model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            fp16=False
        )
        
        transcribed_text = result["text"].strip()
        
        detected_language = result.get("language", "unknown")
        segments = result.get("segments", [])
        num_segments = len(segments)
        
        print(f"Transcription complete! Detected language: {detected_language}")
        
        metadata = {
            "duration_seconds": round(duration_seconds, 2),
            "duration_minutes": round(duration_minutes, 2),
            "file_size_kb": round(file_size_kb, 2),
            "num_segments": num_segments,
            "language_detected": detected_language,
            "extraction_method": "whisper",
            "model_used": "base"
        }
        
        return transcribed_text, metadata
        
    except FileNotFoundError:
        raise Exception(f"Audio file not found: {audio_path}")
    except Exception as e:
        raise Exception(f"Audio transcription failed: {str(e)}")