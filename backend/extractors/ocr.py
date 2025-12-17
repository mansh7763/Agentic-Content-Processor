import pytesseract
from PIL import Image
from typing import Dict, Tuple
import re


def extract_text_from_image(image_path: str) -> Tuple[str, Dict]:
    """
    Extract text from an image using OCR
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (extracted_text, metadata)
    """
    try:
        # Open image
        image = Image.open(image_path)
        
        # Perform OCR with confidence data
        ocr_data = pytesseract.image_to_data(
            image, 
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text
        extracted_text = pytesseract.image_to_string(image)
        
        # Calculate average confidence
        confidences = [
            int(conf) for conf in ocr_data['conf'] 
            if conf != '-1'
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Clean text
        cleaned_text = clean_ocr_text(extracted_text)
        
        metadata = {
            "ocr_confidence": round(avg_confidence, 2),
            "image_size": image.size,
            "words_detected": len([w for w in ocr_data['text'] if w.strip()]),
            "extraction_method": "pytesseract"
        }
        
        return cleaned_text, metadata
        
    except Exception as e:
        raise Exception(f"OCR extraction failed: {str(e)}")


def clean_ocr_text(text: str) -> str:
    """Clean OCR text by removing artifacts and fixing common issues"""
    if not text:
        return ""
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def detect_code_in_text(text: str) -> Dict:
    """
    Detect if text contains code and identify the language
    
    Returns:
        Dict with is_code (bool) and language (str or None)
    """
    code_indicators = {
        'python': [r'def\s+\w+\s*\(', r'import\s+\w+', r'class\s+\w+', r'print\s*\('],
        'javascript': [r'function\s+\w+\s*\(', r'const\s+\w+\s*=', r'let\s+\w+\s*=', r'=>'],
        'java': [r'public\s+class', r'private\s+\w+', r'System\.out\.println'],
        'cpp': [r'#include\s*<', r'int\s+main\s*\(', r'std::'],
        'c': [r'#include\s*<', r'int\s+main\s*\(', r'printf\s*\('],
    }
    
    detected_language = None
    max_matches = 0
    
    for language, patterns in code_indicators.items():
        matches = sum(1 for pattern in patterns if re.search(pattern, text))
        if matches > max_matches:
            max_matches = matches
            detected_language = language
    
    # Consider it code if at least 2 patterns match
    is_code = max_matches >= 2
    
    # Additional heuristic: check for common code structures
    if not is_code:
        code_chars = ['{', '}', ';', '()', '[]']
        code_char_count = sum(text.count(char) for char in code_chars)
        if code_char_count > len(text) * 0.1:  # More than 10% code characters
            is_code = True
    
    return {
        "is_code": is_code,
        "language": detected_language,
        "confidence": min(max_matches / 4, 1.0) if is_code else 0.0
    }