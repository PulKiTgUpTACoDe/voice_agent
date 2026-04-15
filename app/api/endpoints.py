from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
import shutil
import os
from typing import Optional

from app.stt.whisper_engine import stt_engine
from app.agents.intent import classify_intent
from app.agents.graph import run_agent
from app.memory.chroma_store import chroma_store
from app.core.config import settings

router = APIRouter()

class TextRequest(BaseModel):
    text: str

class AgentResponse(BaseModel):
    stt_text: Optional[str] = None
    intent: dict
    output: str

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribes an uploaded audio file using STT engine."""
    file_path = os.path.join(settings.OUTPUT_DIR, f"temp_{file.filename}")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        text = stt_engine.transcribe(file_path)
        return {"text": text}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/classify")
async def classify_text(req: TextRequest):
    """Test endpoint for intent classification."""
    result = classify_intent(req.text)
    return result.model_dump()

@router.post("/agent-run", response_model=AgentResponse)
async def full_agent_run(file: UploadFile = File(None), text: str = None):
    """
    Run the full agent pipeline. Provide either an audio file or raw text.
    """
    input_text = ""
    
    # 1. Pipeline: Audio -> STT
    if file:
        file_path = os.path.join(settings.OUTPUT_DIR, f"temp_{file.filename}")
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            input_text = stt_engine.transcribe(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    elif text:
        input_text = text
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'file' or 'text'")

    if "Error" in input_text or not input_text.strip():
        raise HTTPException(status_code=400, detail=f"Invalid transcription: {input_text}")
    
    # 2. Pipeline: Run Agent (Intent -> Tool -> Action)
    try:
        result = run_agent(input_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
        
    # 3. Save to memory
    chroma_store.add_interaction(user_text=input_text, agent_text=result["output"], metadata={"intent": result["intent"].get("intent", "UNKNOWN")})

    return AgentResponse(
        stt_text=input_text if file else None,
        intent=result["intent"],
        output=result["output"]
    )
