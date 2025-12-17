from typing import Dict
from langchain.prompts import ChatPromptTemplate
from backend.llm.config import get_llm


def summarize_text(text: str, context: str = "") -> Dict:
    """
    Summarize text in three formats:
    1. One-line summary
    2. Three bullet points
    3. Five-sentence summary
    
    Args:
        text: Text to summarize
        context: Additional context about the text
        
    Returns:
        Dict with all three summary formats
    """
    llm = get_llm(temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at creating clear, concise summaries.
You must provide exactly three types of summaries:
1. A single-line summary (one sentence, max 20 words)
2. Three bullet points highlighting key points
3. A five-sentence detailed summary

Format your response EXACTLY as follows:
ONE-LINE: [your one-line summary]

BULLETS:
• [bullet point 1]
• [bullet point 2]
• [bullet point 3]

FIVE-SENTENCES:
[sentence 1] [sentence 2] [sentence 3] [sentence 4] [sentence 5]"""),
        ("user", """Text to summarize:
{text}

{context}

Provide the three summary formats.""")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "text": text[:4000],  # Limit text length
            "context": f"Context: {context}" if context else ""
        })
        
        # Parse the response
        content = response.content
        
        result = parse_summary_response(content)
        result["success"] = True
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "one_liner": "Error generating summary",
            "bullets": ["Error occurred"],
            "five_sentences": "An error occurred while generating the summary."
        }


def parse_summary_response(content: str) -> Dict:
    """Parse the structured summary response"""
    lines = content.strip().split('\n')
    
    one_liner = ""
    bullets = []
    five_sentences = ""
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("ONE-LINE:"):
            one_liner = line.replace("ONE-LINE:", "").strip()
            current_section = "one_liner"
        elif line.startswith("BULLETS:"):
            current_section = "bullets"
        elif line.startswith("FIVE-SENTENCES:"):
            current_section = "five_sentences"
        elif line.startswith("•") or line.startswith("-") or line.startswith("*"):
            if current_section == "bullets":
                bullet = line.lstrip("•-*").strip()
                if bullet:
                    bullets.append(bullet)
        elif line and current_section == "five_sentences":
            five_sentences += line + " "
    
    # Fallback: if parsing failed, try to extract from content
    if not one_liner:
        # Take first sentence
        sentences = content.split('.')
        one_liner = sentences[0].strip() if sentences else "Summary not available"
    
    if not bullets:
        # Split content into sentences and take first 3
        sentences = [s.strip() + '.' for s in content.split('.') if s.strip()]
        bullets = sentences[:3] if len(sentences) >= 3 else ["Summary not available"]
    
    if not five_sentences:
        # Take first 5 sentences
        sentences = [s.strip() + '.' for s in content.split('.') if s.strip()]
        five_sentences = ' '.join(sentences[:5]) if len(sentences) >= 5 else content[:500]
    
    return {
        "one_liner": one_liner.strip(),
        "bullets": bullets[:3],  # Ensure exactly 3
        "five_sentences": five_sentences.strip()
    }