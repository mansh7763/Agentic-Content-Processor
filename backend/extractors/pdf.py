import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from typing import Dict, Tuple
import os


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, Dict]:
    try:
        print(f"Processing PDF: {pdf_path}")
        
        text, metadata = extract_text_directly(pdf_path)
        
        if len(text.strip()) < 50:
            print("Direct extraction yielded minimal text. Trying OCR...")
            text, ocr_metadata = extract_with_ocr(pdf_path)
            
            metadata.update(ocr_metadata)
            metadata["extraction_method"] = "ocr_fallback"
        else:
            print(f"Direct extraction successful! Extracted {len(text)} characters")
        
        return text, metadata
        
    except FileNotFoundError:
        raise Exception(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")


def extract_text_directly(pdf_path: str) -> Tuple[str, Dict]:
    text_content = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            print(f"PDF has {num_pages} pages. Extracting text...")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text:
                    text_content.append(f"\n--- Page {page_num + 1} ---\n")
                    text_content.append(page_text)
        
        full_text = ''.join(text_content)
        
        file_size_kb = os.path.getsize(pdf_path) / 1024.0
        
        metadata = {
            "num_pages": num_pages,
            "extraction_method": "direct",
            "file_size_kb": round(file_size_kb, 2),
            "characters_extracted": len(full_text)
        }
        
        return full_text, metadata
        
    except Exception as e:
        raise Exception(f"Direct PDF extraction failed: {str(e)}")


def extract_with_ocr(pdf_path: str) -> Tuple[str, Dict]:
    try:
        print("Converting PDF pages to images...")
        
        images = convert_from_path(pdf_path, dpi=300)  # Higher DPI = better quality
        
        print(f"Performing OCR on {len(images)} pages...")
        
        text_content = []
        total_confidence = 0
        
        for i, image in enumerate(images):
            print(f"Processing page {i + 1}/{len(images)}...")
            
            page_text = pytesseract.image_to_string(image)
            
            text_content.append(f"\n--- Page {i + 1} ---\n")
            text_content.append(page_text)
            
            try:
                ocr_data = pytesseract.image_to_data(
                    image, 
                    output_type=pytesseract.Output.DICT
                )
                
                confidences = [
                    int(conf) for conf in ocr_data['conf'] 
                    if conf != '-1'
                ]
                
                if confidences:
                    page_confidence = sum(confidences) / len(confidences)
                    total_confidence += page_confidence
                    
            except Exception as e:
                print(f"Warning: Could not calculate confidence for page {i + 1}")
        
        full_text = ''.join(text_content)
        
        avg_confidence = total_confidence / len(images) if images else 0
        
        metadata = {
            "num_pages": len(images),
            "ocr_confidence": round(avg_confidence, 2),
            "extraction_method": "ocr",
            "characters_extracted": len(full_text)
        }
        
        print(f"OCR complete! Average confidence: {avg_confidence:.2f}%")
        
        return full_text, metadata
        
    except Exception as e:
        raise Exception(f"OCR extraction failed: {str(e)}")