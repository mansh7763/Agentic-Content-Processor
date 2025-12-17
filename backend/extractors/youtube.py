from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import re
from typing import Dict, Tuple, Optional


def extract_youtube_video_id(url_or_text: str) -> Optional[str]:
    """
    Extract YouTube video ID from URL or text
    
    Supports formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - youtube.com/watch?v=VIDEO_ID
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_text)
        if match:
            return match.group(1)
    
    return None


def extract_youtube_transcript(video_id: str) -> Tuple[str, Dict]:
    """
    Fetch transcript from YouTube video
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Tuple of (transcript_text, metadata)
    """
    try:
        # Fetch transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all transcript segments
        full_transcript = ' '.join([entry['text'] for entry in transcript_list])
        
        # Calculate duration
        if transcript_list:
            duration_seconds = transcript_list[-1]['start'] + transcript_list[-1]['duration']
        else:
            duration_seconds = 0
        
        metadata = {
            "video_id": video_id,
            "duration_seconds": round(duration_seconds, 2),
            "duration_minutes": round(duration_seconds / 60, 2),
            "num_segments": len(transcript_list),
            "extraction_method": "youtube_transcript_api"
        }
        
        return full_transcript, metadata
        
    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video")
    except NoTranscriptFound:
        raise Exception("No transcript found for this video")
    except Exception as e:
        raise Exception(f"YouTube transcript extraction failed: {str(e)}")


def detect_youtube_url(text: str) -> bool:
    """Check if text contains a YouTube URL"""
    youtube_patterns = [
        r'youtube\.com/watch',
        r'youtu\.be/',
        r'youtube\.com/embed',
        r'youtube\.com/v/'
    ]
    
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in youtube_patterns)