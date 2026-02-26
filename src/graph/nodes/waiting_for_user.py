from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from state_types.state import ChatState
from utils.utils import load_prompt


def make_waiting_for_user_node(*, llm: Any, prompt_dir: Path) -> Callable[[ChatState], Command]:
    """
    Factory for the waiting_for_user node.
    """

    def waiting_for_user_node(state: ChatState) -> Command:
        missing_fields = state.get("missing_fields", []) or []

        prompt = f"""
        {load_prompt('12_waiting_for_user_agent.md', prompts_dir=str(prompt_dir))}

        MISSING FIELDS: {missing_fields}
        """.strip()

        result = llm.invoke([SystemMessage(content=prompt)])

        return Command(
            update={
                **state,
                "messages": (state.get("messages", []) or []) + [AIMessage(content=result.content)],
            },
            goto=END,
        )

    return waiting_for_user_node