import os
from langchain.tools import tool
from app.core.config import settings

@tool
def create_file(filename: str, content: str) -> str:
    """
    Create a new file with the specified text content.
    The file will ALWAYS be created securely in the protected output directory.
    Do not specify full paths, only the filename.
    """
    # Sanitize inputs to avoid path traversal
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(settings.OUTPUT_DIR, safe_filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: File '{safe_filename}' created safely in output directory."
    except Exception as e:
        return f"Error creating file: {str(e)}"

@tool
def write_code(instruction: str, language: str = "python") -> str:
    """
    Generates a code snippet matching the user instruction in the requested language.
    Does NOT save to a file automatically, just returns the code.
    """
    # This tool could simply return the prompt to the language model again, 
    # but since this is an LLM agent, usually the LLM itself generates the code as the final response
    # or passes it to the `create_file` tool.
    # To keep it simple, we wrap returning instructions for clarity in graphing.
    return f"Code written for {instruction} in {language}."

@tool
def summarize(text: str) -> str:
    """
    Summarize the provided text.
    """
    # In LangGraph we might pass this back to an LLM node or use a quick local prompt.
    return f"Summary requested for text of length {len(text)}. The LLM will provide the summary."

@tool
def general_chat(query: str) -> str:
    """
    Execute standard general conversation interaction.
    """
    return "Chat executed."
