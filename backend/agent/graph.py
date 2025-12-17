from langgraph.graph import StateGraph, END
from backend.agent.state import AgentState
from backend.agent.nodes import (
    extract_content_node,
    classify_intent_node,
    execute_task_node,
    ask_followup_node
)


def should_ask_followup(state: AgentState) -> str:
    """Decide whether to ask follow-up question or proceed with task"""
    if state['needs_clarification'] or state['confidence'] < 0.7:
        return "ask_followup"
    return "execute_task"


def create_agent_workflow() -> StateGraph:
    """Create the LangGraph workflow for the agent"""
    
    # Initialize workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_content", extract_content_node)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("ask_followup", ask_followup_node)
    workflow.add_node("execute_task", execute_task_node)
    
    # Set entry point
    workflow.set_entry_point("extract_content")
    
    # Add edges
    workflow.add_edge("extract_content", "classify_intent")
    
    # Conditional edge based on clarity
    workflow.add_conditional_edges(
        "classify_intent",
        should_ask_followup,
        {
            "ask_followup": "ask_followup",
            "execute_task": "execute_task"
        }
    )
    
    # Terminal edges
    workflow.add_edge("ask_followup", END)
    workflow.add_edge("execute_task", END)
    
    return workflow.compile()


def process_followup_response(
    original_state: AgentState,
    followup_response: str
) -> AgentState:
    """
    Process follow-up response and update task
    
    Args:
        original_state: The state when follow-up was asked
        followup_response: User's clarification response
        
    Returns:
        Updated state with new task
    """
    from backend.agent.state import TaskType
    from backend.llm.config import get_llm
    from langchain.prompts import ChatPromptTemplate
    
    llm = get_llm(temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Based on the user's clarification, determine the task they want.

Tasks:
- SUMMARIZE
- SENTIMENT
- CODE_EXPLAIN
- EXTRACT
- QA

Respond with just the task name."""),
        ("user", """Original content type: {input_type}
Original question: {clarification_question}
User's response: {response}

What task do they want?""")
    ])
    
    chain = prompt | llm
    
    try:
        result = chain.invoke({
            "input_type": original_state['input_type'],
            "clarification_question": original_state.get('clarification_question', ''),
            "response": followup_response
        })
        
        task_str = result.content.strip().upper()
        
        # Map to task
        task_mapping = {
            "SUMMARIZE": TaskType.SUMMARIZE,
            "SENTIMENT": TaskType.SENTIMENT,
            "CODE_EXPLAIN": TaskType.CODE_EXPLAIN,
            "EXTRACT": TaskType.EXTRACT,
            "QA": TaskType.QA
        }
        
        detected_task = task_mapping.get(task_str, TaskType.QA)
        
        # Update state
        original_state['detected_task'] = detected_task.value
        original_state['confidence'] = 1.0
        original_state['needs_clarification'] = False
        original_state['user_goal'] = followup_response
        
        # Execute the task
        return execute_task_node(original_state)
        
    except Exception as e:
        original_state['errors'].append(f"Follow-up processing error: {str(e)}")
        return original_state