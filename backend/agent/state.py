from typing import TypedDict, Optional, List, Any, Dict
from enum import Enum


class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    AUDIO = "audio"
    YOUTUBE = "youtube"


class TaskType(str, Enum):
    SUMMARIZE = "summarize"
    SENTIMENT = "sentiment"
    CODE_EXPLAIN = "code_explain"
    EXTRACT = "extract"
    QA = "qa"
    YOUTUBE_TRANSCRIPT = "youtube_transcript"
    UNCLEAR = "unclear"


class AgentState(TypedDict):
    """State for the agent workflow"""
    
    # Input information
    input_type: str  # Type of input received
    raw_input: Any  # Original input (text, file path, etc.)
    file_path: Optional[str]  # Path to uploaded file if any
    
    # Extracted content
    extracted_text: str  # Text extracted from any input type
    extraction_metadata: Dict  # OCR confidence, duration, etc.
    
    # Intent and planning
    user_goal: Optional[str]  # What user wants to do
    detected_task: Optional[str]  # Detected task type
    confidence: float  # Confidence in intent detection (0-1)
    needs_clarification: bool  # Whether follow-up needed
    clarification_question: Optional[str]  # The question to ask
    
    # Execution
    task_plan: Optional[str]  # Plan for executing the task
    result: Optional[Dict]  # Final result
    
    # Conversation management
    conversation_history: List[Dict]  # Chat history
    current_step: str  # Current step in the workflow
    
    # Error handling
    errors: List[str]  # Any errors encountered
    warnings: List[str]  # Any warnings


def create_initial_state(
    input_type: str,
    raw_input: Any,
    file_path: Optional[str] = None
) -> AgentState:
    """Create initial state for the workflow"""
    return AgentState(
        input_type=input_type,
        raw_input=raw_input,
        file_path=file_path,
        extracted_text="",
        extraction_metadata={},
        user_goal=None,
        detected_task=None,
        confidence=0.0,
        needs_clarification=False,
        clarification_question=None,
        task_plan=None,
        result=None,
        conversation_history=[],
        current_step="start",
        errors=[],
        warnings=[]
    )