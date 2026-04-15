from typing import Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
import json
import re

class IntentSchema(BaseModel):
    intent: Literal["CREATE_FILE", "WRITE_CODE", "SUMMARIZE", "GENERAL_CHAT"] = Field(
        description="The classified intent of the user's message."
    )
    parameters: dict = Field(
        default_factory=dict,
        description="Extracted parameters (e.g., filename, file_content, instruction, code_language, text_to_summarize)."
    )

def get_llm():
    """Retrieve the base LLM for the application."""
    return ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.0
    )

def classify_intent(text: str) -> IntentSchema:
    """
    Classify user text into one of the known intents using structured output.
    Uses fallback Regex/JSON parsing if standard structured output fails with the local model.
    """
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intent classification system.
You MUST output ONLY valid JSON matching the following schema. Do NOT wrap in markdown blocks, just raw JSON.
Schema:
{{
  "intent": "CREATE_FILE" | "WRITE_CODE" | "SUMMARIZE" | "GENERAL_CHAT",
  "parameters": {{
     // for CREATE_FILE: "filename", "file_content" (if provided)
     // for WRITE_CODE: "instruction", "code_language"
     // for SUMMARIZE: "text_to_summarize"
  }}
}}
"""),
        ("user", "Input: {input}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"input": text})
    
    content = result.content
    try:
        # Try to clean out markdown json blocks if added
        content_clean = re.sub(r'```json\n|\n```|```', '', content).strip()
        parsed = json.loads(content_clean)
        return IntentSchema(**parsed)
    except Exception as e:
        # Fallback to general chat if parsing fails
        return IntentSchema(
            intent="GENERAL_CHAT", 
            parameters={"query": text, "error": str(e)}
        )
