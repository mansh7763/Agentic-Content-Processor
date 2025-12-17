from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import shutil
from pathlib import Path

from backend.agent.graph import create_agent_workflow, process_followup_response
from backend.agent.state import create_initial_state, InputType

app = FastAPI(title="Agentic Content Processor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

workflow = create_agent_workflow()

session_states = {}


class TextInput(BaseModel):
    text: str
    session_id: Optional[str] = None


class FollowUpInput(BaseModel):
    session_id: str
    response: str


@app.get("/")
async def root():
    return {
        "message": "Agentic Content Processor API",
        "version": "1.0.0",
        "endpoints": {
            "POST /process/text": "Process text input",
            "POST /process/file": "Process file upload (image/pdf/audio)",
            "POST /followup": "Respond to follow-up question",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/process/text")
async def process_text(input_data: TextInput):
    """Process text input"""
    try:
        # Create initial state
        state = create_initial_state(
            input_type=InputType.TEXT,
            raw_input=input_data.text
        )
        
        # Run workflow
        result_state = workflow.invoke(state)
        
        # Check if follow-up needed
        if result_state.get('needs_clarification'):
            # Store state for follow-up
            session_id = generate_session_id()
            session_states[session_id] = result_state
            
            return JSONResponse({
                "status": "needs_clarification",
                "session_id": session_id,
                "question": result_state.get('clarification_question'),
                "extracted_text": result_state.get('extracted_text', '')[:500],
                "metadata": result_state.get('extraction_metadata', {})
            })
        
        # Return result
        return JSONResponse({
            "status": "success",
            "result": result_state.get('result'),
            "extracted_text": result_state.get('extracted_text', '')[:500],
            "metadata": result_state.get('extraction_metadata', {}),
            "task": result_state.get('detected_task'),
            "confidence": result_state.get('confidence')
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/file")
async def process_file(file: UploadFile = File(...)):
    """Process uploaded file (image, pdf, or audio)"""
    try:
        # Determine input type from file extension
        file_ext = file.filename.split('.')[-1].lower()
        
        input_type_mapping = {
            'jpg': InputType.IMAGE,
            'jpeg': InputType.IMAGE,
            'png': InputType.IMAGE,
            'pdf': InputType.PDF,
            'mp3': InputType.AUDIO,
            'wav': InputType.AUDIO,
            'm4a': InputType.AUDIO
        }
        
        if file_ext not in input_type_mapping:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}"
            )
        
        input_type = input_type_mapping[file_ext]
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create initial state
        state = create_initial_state(
            input_type=input_type,
            raw_input=None,
            file_path=str(file_path)
        )
        
        # Run workflow
        result_state = workflow.invoke(state)
        
        # Check if follow-up needed
        if result_state.get('needs_clarification'):
            session_id = generate_session_id()
            session_states[session_id] = result_state
            
            return JSONResponse({
                "status": "needs_clarification",
                "session_id": session_id,
                "question": result_state.get('clarification_question'),
                "extracted_text": result_state.get('extracted_text', '')[:500],
                "metadata": result_state.get('extraction_metadata', {})
            })
        
        # Return result
        return JSONResponse({
            "status": "success",
            "result": result_state.get('result'),
            "extracted_text": result_state.get('extracted_text', '')[:500],
            "metadata": result_state.get('extraction_metadata', {}),
            "task": result_state.get('detected_task'),
            "confidence": result_state.get('confidence')
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/followup")
async def handle_followup(input_data: FollowUpInput):
    """Handle follow-up response"""
    try:
        # Get stored state
        if input_data.session_id not in session_states:
            raise HTTPException(status_code=404, detail="Session not found")
        
        original_state = session_states[input_data.session_id]
        
        # Process follow-up
        result_state = process_followup_response(
            original_state,
            input_data.response
        )
        
        # Clean up session
        del session_states[input_data.session_id]
        
        return JSONResponse({
            "status": "success",
            "result": result_state.get('result'),
            "task": result_state.get('detected_task'),
            "confidence": result_state.get('confidence')
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_session_id() -> str:
    """Generate unique session ID"""
    import uuid
    return str(uuid.uuid4())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)