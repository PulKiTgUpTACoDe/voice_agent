from pydantic import BaseModel
import os
from pathlib import Path

class Settings(BaseModel):
    # Application settings
    APP_NAME: str = "Voice Controlled Local AI Agent"
    
    # Model configuration
    LLM_MODEL: str = "llama3" # Default Ollama model
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # STT configuration
    WHISPER_MODEL_SIZE: str = "base"
    
    # Paths
    BASE_DIR: str = str(Path(__file__).parent.parent.parent)
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    DB_DIR: str = os.path.join(BASE_DIR, "chroma_db")

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.DB_DIR, exist_ok=True)

settings = Settings()
