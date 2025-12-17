from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def get_llm(temperature: float = 0.1, model_name: str = None):
    """
    Get Groq LLM instance
    
    Args:
        temperature: Temperature for generation (0-1)
        model_name: Model name to use (defaults to env variable)
        
    Returns:
        ChatGroq instance
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=2000
    )


def get_structured_llm(temperature: float = 0.1):
    """Get LLM configured for structured output"""
    return get_llm(temperature=temperature)