from __future__ import annotations

from typing import Any, Callable, Dict

from langgraph.types import Command

from state_types.state import ChatState
from utils.utils import mark_agent_completed
from predict.predict import predict_from_profile


def make_predict_node(
    *,
    model: Any,
    scaler: Any,
    completed_agent_name: str = "predictor",
) -> Callable[[ChatState], Command]:


    def predict_node(state: ChatState) -> Command:
        profile = state.get("profile", {}) or {}
        if not profile:
            raise ValueError("Profile data is missing in state.")

        pred_result = predict_from_profile(
            profile,  # type: ignore
            model=model,
            scaler=scaler,
        )

        update: Dict[str, Any] = {
            **state,
            "prediction": pred_result,
            **mark_agent_completed(state, completed_agent_name),
        }

        return Command(update=update, goto="supervisor")

    return predict_node