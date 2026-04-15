# Voice-Controlled Local AI Agent

A production-ready voice-controlled AI system that runs locally. It accepts audio input, transcribes it using robust STT (faster-whisper), classifies intent with a local LLM via Ollama, executes tools safely via LangGraph orchestration, and stores interactions in ChromaDB memory. The system is exposed through FastAPI and accessed via a Streamlit UI.

## Features
- **Local STT**: `faster-whisper` for fast local inference.
- **Local LLM**: LangChain integration with Ollama.
- **Intent Classification & Graph Routing**: Powered by LangGraph and structured output parsing.
- **ChromaDB**: For contextual conversational memory.
- **Safety First**: Output file creation is restricted to the `/output` folder.

## Setup Requirements

1. **Python 3.10+**
2. **Ollama**: Must be installed and running.
   - Download from https://ollama.com.
   - Once installed, pull the Llama 3 model:
     ```bash
     ollama pull llama3
     ```
3. Create a python virtual environment, activate it, and install requirements:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   pip install -r requirements.txt
   ```

## Running the Application

You can start the backend and frontend separately or use the provided startup script.

**1. FastAPI Backend:**
```bash
uvicorn main:app --reload --port 8000
```

**2. Streamlit UI:**
```bash
streamlit run frontend/streamlit_app.py
```

## Structure
- `app/api`: HTTP Routing.
- `app/stt`: Transcription system.
- `app/agents`: LangGraph logic, Intent Classification.
- `app/tools`: Custom tools for actions like code gen, file writing.
- `app/memory`: ChromaDB persistent state.
- `frontend`: Streamlit UI.

## Constraints
All generated files and scripts are safely written strictly to the `output/` directory located inside this project folder.
