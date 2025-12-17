from typing import Dict
from langchain.prompts import ChatPromptTemplate
from backend.llm.config import get_llm


def answer_question(question: str, context: str = "") -> Dict:
    """
    Answer a question, optionally with context
    
    Args:
        question: The question to answer
        context: Optional context from extracted content
        
    Returns:
        Dict with answer
    """
    llm = get_llm(temperature=0.3)
    
    if context:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on provided context.
If the answer is in the context, provide a clear and accurate response.
If the answer is not in the context, say so and provide general knowledge if appropriate.
Be concise, friendly, and helpful."""),
            ("user", """Context:
{context}

Question: {question}

Answer:""")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful, friendly assistant.
Provide clear, accurate, and conversational responses.
Be concise but thorough."""),
            ("user", "{question}")
        ])
    
    chain = prompt | llm
    
    try:
        if context:
            response = chain.invoke({
                "question": question,
                "context": context[:3000]  # Limit context length
            })
        else:
            response = chain.invoke({"question": question})
        
        return {
            "success": True,
            "answer": response.content.strip()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": "I apologize, but I encountered an error while processing your question."
        }


def extract_action_items(text: str) -> Dict:
    """
    Extract action items from meeting notes or similar text
    
    Args:
        text: Text to extract action items from
        
    Returns:
        Dict with list of action items
    """
    llm = get_llm(temperature=0.2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting action items from meeting notes and documents.
Action items are tasks, to-dos, or decisions that require follow-up.

Extract ALL action items and format them as a numbered list.
For each action item, include:
- What needs to be done
- Who is responsible (if mentioned)
- Deadline (if mentioned)

If no clear action items exist, say "No specific action items found."""),
        ("user", """Extract action items from this text:

{text}

Action items:""")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"text": text[:4000]})
        content = response.content.strip()
        
        # Parse action items into a list
        action_items = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                cleaned = line.lstrip('0123456789.-•*').strip()
                if cleaned:
                    action_items.append(cleaned)
        
        if not action_items and content:
            # Fallback: treat the whole response as one item
            action_items = [content]
        
        return {
            "success": True,
            "action_items": action_items,
            "count": len(action_items)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action_items": [],
            "count": 0
        }