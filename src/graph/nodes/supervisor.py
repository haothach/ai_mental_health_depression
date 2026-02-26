from __future__ import annotations

from pathlib import Path
from typing import Callable, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.types import Command

from state_types.state import ChatState, Intent, IntentClassifierOutput
from utils.utils import (
    get_last_user_message,
    load_prompt,
    create_supervisor_update_with_workflow_init,
)


def detect_intent(messages: list, *, llm_super: ChatOpenAI, prompt_dir: Path) -> Intent:
    """Detect intent from last user message using the supervisor LLM."""
    user_text = get_last_user_message(messages)
    if not user_text:
        return "UNKNOWN"

    llm_intent = llm_super.with_structured_output(IntentClassifierOutput)

    prompt = f"""{load_prompt("10_router_agent.md", prompts_dir=str(prompt_dir))}
        USER MESSAGE: "{user_text}" 
    """

    out = cast(
        IntentClassifierOutput,
        llm_intent.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=user_text),
            ]
        ),
    )
    return out.intent


def make_supervisor_node(*, llm_super: ChatOpenAI, prompt_dir: Path) -> Callable[[ChatState], Command]:
    """
    Factory returning a LangGraph node function with injected dependencies.
    """

    def supervisor_node(state: ChatState) -> Command:
        messages = state.get("messages", []) or []
        pending_agents = state.get("pending_agents", []) or []
        completed_agents = state.get("completed_agents", []) or []
        intent = state.get("intent")

        # If intent not set yet, classify and initialize workflow
        if not intent:
            detected_intent = detect_intent(messages, llm_super=llm_super, prompt_dir=prompt_dir)

            # Direct paths
            if detected_intent in ["UNKNOWN", "DIRECT_RESPONSE"]:
                update = {
                    **state,
                    "intent": detected_intent,
                    "messages": messages
                    + [AIMessage(content=f"[Supervisor] Detected intent: '{detected_intent}'")],
                }
                return Command(update=update, goto="direct_responder")

            workflow_update, resolved_intent, new_pending_agents = create_supervisor_update_with_workflow_init(
                detected_intent, state
            )

            if not workflow_update:
                return Command(
                    update={
                        **state,
                        "messages": messages
                        + [AIMessage(content=f"[Supervisor] Intent '{detected_intent}' not supported")],
                    },
                    goto=END,
                )

            if not new_pending_agents:
                return Command(
                    update={
                        **state,
                        **workflow_update,
                        "intent": None,
                        "pending_agents": [],
                        "completed_agents": [],
                    },
                    goto=END,
                )

            next_agent = new_pending_agents[0]
            debug_msg = (
                f"[Supervisor] → '{next_agent}' | Pending: {new_pending_agents} | Completed: {completed_agents}"
            )
            return Command(
                update={
                    **state,
                    **workflow_update,
                    "intent": resolved_intent,
                    "messages": messages + [AIMessage(content=debug_msg)],
                },
                goto=next_agent,
            )

        # If workflow finished
        if len(pending_agents) == 0:
            return Command(
                update={
                    **state,
                    "intent": None,
                    "pending_agents": [],
                    "completed_agents": [],
                },
                goto=END,
            )

        # Continue workflow
        next_agent = pending_agents[0]
        debug_msg = f"[Supervisor] → '{next_agent}' | Pending: {pending_agents} | Completed: {completed_agents}"
        return Command(
            update={
                **state,
                "messages": messages + [AIMessage(content=debug_msg)],
            },
            goto=next_agent,
        )

    return supervisor_node