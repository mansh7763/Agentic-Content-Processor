from typing import Dict
from langchain.prompts import ChatPromptTemplate
from backend.llm.config import get_llm


def analyze_sentiment(text: str) -> Dict:
    llm = get_llm(temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert sentiment analyzer.

Analyze the sentiment of the text and provide your response in this EXACT format:

SENTIMENT: [positive/negative/neutral]
CONFIDENCE: [0.0-1.0]
JUSTIFICATION: [one clear sentence explaining why you classified it this way]

Important guidelines:
- Use ONLY these three sentiment labels: positive, negative, or neutral
- Confidence should be a number between 0.0 (not confident) and 1.0 (very confident)
- Justification should be a single, clear sentence explaining your reasoning
- Be objective and focus on the actual sentiment expressed in the text

Example response:
SENTIMENT: positive
CONFIDENCE: 0.85
JUSTIFICATION: The text expresses enthusiasm and satisfaction with clear positive language and emotional words."""),
        ("user", "Analyze the sentiment of this text:\n\n{text}")
    ])
    
    chain = prompt | llm
    
    try:
        print("Analyzing sentiment with LLM...")
        response = chain.invoke({"text": text[:2000]})  # Limit text length
        
        content = response.content
        result = parse_sentiment_response(content)
        
        result["success"] = True
        
        print(f"Sentiment analysis complete! Detected: {result['label']} ({result['confidence']})")
        return result
        
    except Exception as e:
        print(f"Error during sentiment analysis: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "label": "neutral",
            "confidence": 0.0,
            "justification": "Error occurred during sentiment analysis"
        }


def parse_sentiment_response(content: str) -> Dict:
    sentiment = "neutral"
    confidence = 0.5
    justification = ""
    
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("SENTIMENT:"):
            sentiment_text = line.replace("SENTIMENT:", "").strip().lower()
            if sentiment_text in ["positive", "negative", "neutral"]:
                sentiment = sentiment_text
            else:
                if "positive" in sentiment_text:
                    sentiment = "positive"
                elif "negative" in sentiment_text:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
        
        elif line.startswith("CONFIDENCE:"):
            conf_text = line.replace("CONFIDENCE:", "").strip()
            try:
                confidence = float(conf_text)
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5
        
        elif line.startswith("JUSTIFICATION:"):
            justification = line.replace("JUSTIFICATION:", "").strip()
    
    if sentiment == "neutral" and not justification:
        content_lower = content.lower()
        
        if "positive" in content_lower and "not" not in content_lower[:content_lower.find("positive")]:
            sentiment = "positive"
        elif "negative" in content_lower and "not" not in content_lower[:content_lower.find("negative")]:
            sentiment = "negative"
        
        justification = content[:200] if not justification else justification
    
    if not justification:
        justification = f"The text appears to have a {sentiment} sentiment based on language analysis."
    
    return {
        "label": sentiment,
        "confidence": round(confidence, 2),
        "justification": justification
    }