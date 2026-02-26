from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from state_types.state import ChatState
from utils.utils import get_last_user_message, load_prompt, mark_agent_completed


def make_direct_response_node(*, llm: Any, prompt_dir: Path) -> Callable[[ChatState], Command]:

    def direct_response_node(state: ChatState) -> Command:
        messages = state.get("messages", []) or []
        user_text = get_last_user_message(messages)

        prompt = f"""
        {load_prompt('13_responder_agent.md', prompts_dir=str(prompt_dir))}
        """.strip()

        result = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=user_text),
            ]
        )

        update: Dict[str, Any] = {
            **state,
            "messages": messages + [AIMessage(content=result.content)],
            **mark_agent_completed(state, "direct_responder"),
            "intent": None,
            "pending_agents": [],
            "completed_agents": [],
        }

        return Command(update=update, goto=END)

    return direct_response_node