from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from app.agents.intent import classify_intent, IntentSchema, get_llm
from app.tools.all_tools import create_file, write_code, summarize, general_chat
import operator
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    input_text: str
    intent: IntentSchema
    output: str
    memory_context: str
    messages: Annotated[Sequence[BaseMessage], operator.add]

def classify_node(state: AgentState):
    """Classifies the intent based on the raw audio transcription."""
    intent_data = classify_intent(state["input_text"])
    logger.info(f"Classified Intent: {intent_data.intent}")
    return {"intent": intent_data}

def execute_tool_node(state: AgentState):
    """Executes the appropriate tool logic, potentially calling the LLM directly."""
    intent = state["intent"]
    llm = get_llm()
    
    output = ""
    
    if intent.intent == "CREATE_FILE":
        filename = intent.parameters.get("filename", "output.txt")
        content = intent.parameters.get("file_content", "")
        # If content isn't provided, use LLM to draft something
        if not content:
            prompt = f"Conversation History:\n{state.get('memory_context', '')}\n\nDraft content for a file named {filename} based on this request: {state['input_text']}"
            content = llm.invoke(prompt).content
        output = create_file.invoke({"filename": filename, "content": content})
        
    elif intent.intent == "WRITE_CODE":
        instruction = intent.parameters.get("instruction", state["input_text"])
        language = intent.parameters.get("code_language", "python")
        prompt = f"Conversation History:\n{state.get('memory_context', '')}\n\nWrite {language} code for: {instruction}. Output ONLY the complete code, no explanations."
        code_content = llm.invoke(prompt).content
        
        # Save automatically to output dir
        filename = intent.parameters.get("filename", f"script.{'py' if language.lower() == 'python' else 'txt'}")
        save_result = create_file.invoke({"filename": filename, "content": code_content})
        output = f"Code generated:\n```\n{code_content}\n```\nResult: {save_result}"
        
    elif intent.intent == "SUMMARIZE":
        text_to_summarize = intent.parameters.get("text_to_summarize", state["input_text"])
        prompt = f"Conversation History:\n{state.get('memory_context', '')}\n\nSummarize the following text:\n{text_to_summarize}"
        summary = llm.invoke(prompt).content
        output = f"Summary: {summary}"
        
    else:
        # GENERAL_CHAT
        prompt = f"Conversation History:\n{state.get('memory_context', '')}\n\nThe user said: {state['input_text']}\nRespond helpfully."
        output = llm.invoke(prompt).content
        
    # Append message interaction
    messages = [HumanMessage(content=state['input_text']), AIMessage(content=output)]
    return {"output": output, "messages": messages}

# Build LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_node)
workflow.add_node("execute", execute_tool_node)

workflow.set_entry_point("classify")
workflow.add_edge("classify", "execute")
workflow.add_edge("execute", END)

agent_app = workflow.compile()

def run_agent(text: str) -> dict:
    """Entry point for running the agent pipeline"""
    from app.memory.chroma_store import chroma_store
    past_context = chroma_store.get_context(n_results=3)
    
    initial_state = {
        "input_text": text,
        "intent": None,
        "output": "",
        "memory_context": past_context,
        "messages": []
    }
    result = agent_app.invoke(initial_state)
    return {
        "intent": result["intent"].model_dump(),
        "output": result["output"]
    }
