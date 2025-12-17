from backend.agent.state import AgentState, TaskType, InputType
from backend.extractors.ocr import extract_text_from_image, detect_code_in_text
from backend.extractors.pdf import extract_text_from_pdf
from backend.extractors.audio import extract_text_from_audio
from backend.extractors.youtube import extract_youtube_transcript, extract_youtube_video_id, detect_youtube_url
from backend.tasks.summarize import summarize_text
from backend.tasks.sentiment import analyze_sentiment
from backend.tasks.code_explain import explain_code
from backend.tasks.qa import answer_question, extract_action_items
from langchain.prompts import ChatPromptTemplate
from backend.llm.config import get_llm


def extract_content_node(state: AgentState) -> AgentState:
    """Extract content from various input types"""
    print(f"[NODE] Extracting content from {state['input_type']}")
    
    try:
        if state['input_type'] == InputType.TEXT:
            # Check if text contains YouTube URL
            text = state['raw_input']
            if detect_youtube_url(text):
                video_id = extract_youtube_video_id(text)
                if video_id:
                    # Extract YouTube transcript
                    transcript, metadata = extract_youtube_transcript(video_id)
                    state['extracted_text'] = transcript
                    state['extraction_metadata'] = metadata
                    state['input_type'] = InputType.YOUTUBE
                else:
                    state['extracted_text'] = text
                    state['extraction_metadata'] = {"method": "direct_text"}
            else:
                state['extracted_text'] = text
                state['extraction_metadata'] = {"method": "direct_text"}
                
        elif state['input_type'] == InputType.IMAGE:
            text, metadata = extract_text_from_image(state['file_path'])
            state['extracted_text'] = text
            state['extraction_metadata'] = metadata
            
            # Check if image contains code
            code_detection = detect_code_in_text(text)
            state['extraction_metadata']['code_detection'] = code_detection
            
        elif state['input_type'] == InputType.PDF:
            text, metadata = extract_text_from_pdf(state['file_path'])
            state['extracted_text'] = text
            state['extraction_metadata'] = metadata
            
        elif state['input_type'] == InputType.AUDIO:
            text, metadata = extract_text_from_audio(state['file_path'])
            state['extracted_text'] = text
            state['extraction_metadata'] = metadata
            
        state['current_step'] = 'content_extracted'
        
    except Exception as e:
        state['errors'].append(f"Content extraction error: {str(e)}")
        state['extracted_text'] = ""
        state['current_step'] = 'extraction_failed'
    
    return state


def classify_intent_node(state: AgentState) -> AgentState:
    """Classify user intent and determine task type"""
    print("[NODE] Classifying intent")
    
    llm = get_llm(temperature=0.1)
    
    # Build context about the input
    context_parts = []
    if state['input_type'] == InputType.IMAGE:
        context_parts.append("User uploaded an image.")
        if state['extraction_metadata'].get('code_detection', {}).get('is_code'):
            context_parts.append("The image contains code.")
    elif state['input_type'] == InputType.PDF:
        context_parts.append(f"User uploaded a PDF with {state['extraction_metadata'].get('num_pages', 0)} pages.")
    elif state['input_type'] == InputType.AUDIO:
        duration = state['extraction_metadata'].get('duration_minutes', 0)
        context_parts.append(f"User uploaded audio ({duration:.1f} minutes).")
    elif state['input_type'] == InputType.YOUTUBE:
        context_parts.append("User provided a YouTube URL.")
    
    context = " ".join(context_parts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intent classifier for an AI agent.
Analyze the extracted content and determine what the user wants to do.

Possible tasks:
1. SUMMARIZE - User wants a summary of the content
2. SENTIMENT - User wants sentiment analysis
3. CODE_EXPLAIN - User wants code explained (if content contains code)
4. EXTRACT - User wants specific information extracted (e.g., action items)
5. QA - User has a question or wants conversational response
6. UNCLEAR - Cannot determine intent confidently

Respond in this EXACT format:
TASK: [task name from above]
CONFIDENCE: [0.0-1.0]
REASONING: [one sentence why]
NEEDS_CLARIFICATION: [yes/no]
CLARIFICATION_QUESTION: [question to ask if needs clarification, otherwise "none"]

Guidelines:
- If content is extracted from audio/pdf/image without clear instruction, confidence should be < 0.6
- If user asks explicit question, task is QA with high confidence
- If "summarize" or "summary" mentioned, task is SUMMARIZE
- If "sentiment" or "feeling" mentioned, task is SENTIMENT
- If code is detected and user says "explain", task is CODE_EXPLAIN
- If asking for "action items" or specific extraction, task is EXTRACT
- Set NEEDS_CLARIFICATION to yes if confidence < 0.7 or task is ambiguous"""),
        ("user", """{context}

Extracted content:
{text}

What does the user want?""")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "context": context,
            "text": state['extracted_text'][:1500]  # Limit text
        })
        
        # Parse response
        content = response.content
        lines = content.strip().split('\n')
        
        task = TaskType.UNCLEAR
        confidence = 0.5
        reasoning = ""
        needs_clarification = True
        clarification_question = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("TASK:"):
                task_str = line.replace("TASK:", "").strip().upper()
                # Map to TaskType
                task_mapping = {
                    "SUMMARIZE": TaskType.SUMMARIZE,
                    "SENTIMENT": TaskType.SENTIMENT,
                    "CODE_EXPLAIN": TaskType.CODE_EXPLAIN,
                    "EXTRACT": TaskType.EXTRACT,
                    "QA": TaskType.QA,
                    "UNCLEAR": TaskType.UNCLEAR
                }
                task = task_mapping.get(task_str, TaskType.UNCLEAR)
                
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except:
                    confidence = 0.5
                    
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
                
            elif line.startswith("NEEDS_CLARIFICATION:"):
                needs_clarification = "yes" in line.lower()
                
            elif line.startswith("CLARIFICATION_QUESTION:"):
                clarification_question = line.replace("CLARIFICATION_QUESTION:", "").strip()
                if clarification_question.lower() == "none":
                    clarification_question = ""
        
        state['detected_task'] = task.value
        state['confidence'] = confidence
        state['user_goal'] = reasoning
        state['needs_clarification'] = needs_clarification
        state['clarification_question'] = clarification_question if clarification_question else generate_fallback_question(task, state)
        state['current_step'] = 'intent_classified'
        
    except Exception as e:
        state['errors'].append(f"Intent classification error: {str(e)}")
        state['needs_clarification'] = True
        state['clarification_question'] = "I extracted the content, but I'm not sure what you'd like me to do with it. Could you clarify?"
        state['current_step'] = 'classification_failed'
    
    return state


def generate_fallback_question(task: TaskType, state: AgentState) -> str:
    """Generate fallback clarification question based on context"""
    if state['input_type'] == InputType.IMAGE:
        if state['extraction_metadata'].get('code_detection', {}).get('is_code'):
            return "I detected code in the image. Would you like me to explain it, or do something else?"
        return "I extracted text from the image. What would you like me to do with it? (e.g., summarize, analyze sentiment, answer a question)"
    
    elif state['input_type'] == InputType.PDF:
        return "I extracted the PDF content. What would you like me to do? (e.g., summarize, find action items, answer a question)"
    
    elif state['input_type'] == InputType.AUDIO:
        return "I transcribed the audio. Would you like a summary, or something else?"
    
    elif state['input_type'] == InputType.YOUTUBE:
        return "I fetched the YouTube transcript. Would you like me to summarize it, or do something else?"
    
    return "What would you like me to do with this content?"


def execute_task_node(state: AgentState) -> AgentState:
    """Execute the determined task"""
    print(f"[NODE] Executing task: {state['detected_task']}")
    
    try:
        task = state['detected_task']
        text = state['extracted_text']
        metadata = state['extraction_metadata']
        
        if task == TaskType.SUMMARIZE.value:
            result = summarize_text(text)
            result['metadata'] = metadata
            state['result'] = result
            
        elif task == TaskType.SENTIMENT.value:
            result = analyze_sentiment(text)
            state['result'] = result
            
        elif task == TaskType.CODE_EXPLAIN.value:
            language = metadata.get('code_detection', {}).get('language')
            result = explain_code(text, language)
            state['result'] = result
            
        elif task == TaskType.EXTRACT.value:
            # Check if user wants action items
            if "action" in state.get('user_goal', '').lower():
                result = extract_action_items(text)
            else:
                result = extract_action_items(text)
            state['result'] = result
            
        elif task == TaskType.QA.value:
            question = state.get('raw_input', '')
            result = answer_question(question, text)
            state['result'] = result
            
        elif task == TaskType.YOUTUBE_TRANSCRIPT.value:
            # Just return the transcript with metadata
            state['result'] = {
                "success": True,
                "transcript": text,
                "metadata": metadata
            }
        
        state['current_step'] = 'task_completed'
        
    except Exception as e:
        state['errors'].append(f"Task execution error: {str(e)}")
        state['result'] = {
            "success": False,
            "error": str(e)
        }
        state['current_step'] = 'execution_failed'
    
    return state


def ask_followup_node(state: AgentState) -> AgentState:
    """Ask follow-up question to clarify intent"""
    print("[NODE] Asking follow-up question")
    
    state['current_step'] = 'awaiting_clarification'
    return state