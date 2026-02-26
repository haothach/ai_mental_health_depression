from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from state_types.state import ChatState, IntakerOutput, UserProfile
from utils.utils import get_last_user_message, load_prompt, mark_agent_completed


def make_intaker_node(*, intaker_llm: Any, prompt_dir: Path) -> Callable[[ChatState], Command]:

    def intaker_node(state: ChatState) -> Command:
        messages = state.get("messages", []) or []
        user_text = get_last_user_message(messages)
        current_profile = state.get("profile", {}) or {}

        prompt = f"""
        {load_prompt('11_intaker_agent.md', prompts_dir=str(prompt_dir))}

        CURRENT PROFILE:
        {current_profile}
        USER INPUT:
        {user_text}
        """.strip()

        response = cast(
            IntakerOutput,
            intaker_llm.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=user_text),
                ]
            ),
        )

        new_profile: Dict[str, Any] = {**current_profile} # type: ignore
        if response.profile:
            for k, v in asdict(response.profile).items():
                if v is not None:
                    new_profile[k] = v

        profile = UserProfile.model_construct(**new_profile)
        missing_fields = profile.get_missing_fields()

        update: Dict[str, Any] = {
            **state,
            "profile": new_profile,
            "missing_fields": missing_fields,
            **(mark_agent_completed(state, "intaker") if not missing_fields else {}),
        }

        return Command(update=update, goto="waiting_for_user" if missing_fields else "supervisor")

    return intaker_node