from langchain_core.messages import HumanMessage
from pathlib import Path
from typing import Annotated, Literal, Optional, List, Dict, Any

from langchain_core.messages import AIMessage
from state_types.state import ChatState, Intent


def get_last_user_message(messages: list) -> str:
    """
    Extract last user message from conversation.
    """
    
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""

def load_prompt(filename: str, prompts_dir: str) -> str:
    """
    Load prompt from file.
    """
    # Try loading from file if directory provided
    if prompts_dir:
        prompt_path = Path(prompts_dir) / filename
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        
    return ""


SESSION_WORKFLOWS: Dict[Intent, Dict[str, Any]] = {
    "PREDICT": {
        "agents": ["intaker", "predictor", "advisor"],
        "description": "Workflow for predicting depression risk based on user input and profile extraction."
    },
    "EXTRACT_PROFILE": {
        "agents": ["intaker"],
        "description": "Workflow for extracting user profile information from conversation."
    },
    "DIRECT_RESPONSE": {
        "agents": ["direct_responder"],
        "description": "Directly responding to user queries without additional processing."
    },
    "RAG_QA": {
        "agents": ["rag_responder"],
        "description": "Workflow for answering questions using Retrieval-Augmented Generation (RAG) techniques."
    },
    "EVALUATE_AND_ADVISE": {
        "agents": ["advisor"],
        "description": "Workflow for evaluating depression risk and providing advice based on user profile."
    },
}


def initialize_workflow(intent: Intent, state: ChatState) -> Dict[str, Any]:
    """
    Initialize workflow for given intent.
    """
    workflow = SESSION_WORKFLOWS.get(intent)
    if not workflow:
        return {}
    
    # Get list of agents not yet completed
    completed = set(state.get("completed_agents", []) or [])
    pending = [agent for agent in workflow["agents"] if agent not in completed]
    
    return {
        "intent": intent,
        "pending_agents": pending,
        "completed_agents": list(completed),
    }


def mark_agent_completed(state: ChatState, agent_name: str) -> Dict[str, Any]:
    """
    Mark agent as completed and update pending list.
    """
    pending = list(state.get("pending_agents", []) or [])
    completed = list(state.get("completed_agents", []) or [])
    
    if agent_name in pending:
        pending.remove(agent_name)
    
    if agent_name not in completed:
        completed.append(agent_name)
    
    return {
        "pending_agents": pending,
        "completed_agents": completed,
    }


def get_next_agent(state: ChatState) -> Optional[str]:
    """
    Determine next agent to run based on workflow state.

    """
    pending_agents = state.get("pending_agents", []) or []
    
    if pending_agents:
        return pending_agents[0]
    
    return None

def create_supervisor_update_with_workflow_init(
    detected_intent: Intent,
    state: ChatState
) -> tuple[Dict[str, Any], Optional[Intent], List[str]]:
    """
    Create update dict when initializing new workflow.
    """
    workflow_init = initialize_workflow(detected_intent, state)
    
    if not workflow_init:
        return {}, None, []
    
    session_intent = detected_intent
    pending_agents = workflow_init.get("pending_agents", [])
    
    update = workflow_init.copy()
    debug_msg = f"[Supervisor] New session: intent={session_intent}, workflow={' → '.join(pending_agents)}"
    update["messages"] = [AIMessage(content=debug_msg)]
    
    return update, session_intent, pending_agents

def get_workflow_description(intent: Intent) -> str:
    """
    Get human-readable description of workflow.
    """
    workflow = SESSION_WORKFLOWS.get(intent)
    if not workflow:
        return "Unknown workflow"
    return workflow.get("description", "Workflow")