from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.types import Command

from state_types.state import ChatState
from utils.utils import get_last_user_message, load_prompt, mark_agent_completed


def build_advice_retrieval_query(user_text: str, profile: dict, prediction: Any) -> str:
    """Build a context-rich query for RAG retrieval (grounding docs). (Ported from demo.ipynb)"""
    pred_label = ""
    pred_score = ""
    if isinstance(prediction, dict):
        pred_label = str(prediction.get("label") or prediction.get("risk_label") or prediction.get("class") or "")
        pred_score = str(
            prediction.get("score") or prediction.get("probability") or prediction.get("risk_score") or ""
        )

    salient_keys = [
        "age",
        "gender",
        "sleep_duration",
        "dietary_habits",
        "study_hours",
        "academic_pressure",
        "study_satisfaction",
        "financial_stress",
        "family_history",
        "suicidal_thoughts",
    ]
    salient = {k: profile.get(k) for k in salient_keys if profile.get(k) is not None}

    safety_focus = ""
    if profile.get("suicidal_thoughts") is True:
        safety_focus = (
            "\nSafety priority: user indicates suicidal thoughts. Retrieve crisis/safety guidance, "
            "when to seek urgent help, safety planning, and how to encourage reaching out."
        )

    return (
        "Task: Retrieve evidence-based guidance passages for a student mental health assistant.\n"
        f"User request: {user_text}\n"
        f"Risk screening signal (NOT diagnosis): label={pred_label}, score={pred_score}\n"
        f"User profile key factors: {json.dumps(salient, ensure_ascii=False)}\n"
        "Need documents covering:\n"
        "- Practical coping steps (sleep, routine, stress, study workload)\n"
        "- When to seek professional help (red flags, escalation)\n"
        "- How to talk about feelings and social support\n"
        f"{safety_focus}\n"
        "Return passages suitable to cite in a short advice message.\n"
    )


def _as_text(x: Any) -> str:
    """Normalize chain output to a string (some LC chains may return dict/message)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return str(x.get("answer") or x.get("text") or x.get("output_text") or x)
    content = getattr(x, "content", None)
    if isinstance(content, str):
        return content
    return str(x)


def make_eva_advise_node(
    *,
    llm: Any,
    prompt_dir: Path,
    rag_retriever: Callable[[str], List[Any]],
    prompt_name: str = "15_evaluate_and_advise_agent.md",
    completed_agent_name: str = "advisor",
) -> Callable[[ChatState], Command]:
    """
    Evaluate & Advise node (ported from demo.ipynb's eva_adv_node).

    Flow:
      - require profile + prediction
      - build rag_query from (user_text, profile, prediction)
      - retrieve docs with rag_retriever(rag_query)
      - stuff docs into {context} and generate answer
      - mark completed + goto supervisor
    """

    def eva_advise_node(state: ChatState) -> Command:
        messages = state.get("messages", []) or []
        user_text = get_last_user_message(messages)
        profile = state.get("profile", {}) or {}
        prediction = state.get("prediction")

        if not profile or not prediction:
            return Command(
                update={
                    **state,
                    "messages": messages
                    + [
                        AIMessage(
                            content="Insufficient data to provide advice. Please complete the intake process."
                        )
                    ],
                    **mark_agent_completed(state, completed_agent_name),
                },
                goto=END,
            )

        rag_query = build_advice_retrieval_query(user_text, profile, prediction) # type: ignore
        retrieved_docs = rag_retriever(rag_query)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt(prompt_name, prompts_dir=str(prompt_dir))),
                (
                    "human",
                    "USER REQUEST:\n{user_query}\n\n"
                    "USER PROFILE:\n{user_profile}\n\n"
                    "PREDICTION (screening only, not diagnosis):\n{prediction}\n\n"
                    "GROUNDED CONTEXT (retrieved docs):\n{context}\n",
                ),
            ]
        )

        stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
        answer = stuff_chain.invoke(
            {
                "user_query": user_text,
                "user_profile": json.dumps(profile, ensure_ascii=False),
                "prediction": json.dumps(prediction, ensure_ascii=False),
                "context": retrieved_docs,
            }
        )

        answer_text = _as_text(answer)

        return Command(
            update={
                **state,
                "rag_advice_query": rag_query,
                "rag_advice_docs": retrieved_docs,
                "messages": messages + [AIMessage(content=answer_text)],
                **mark_agent_completed(state, completed_agent_name),
            },
            goto="supervisor",
        )

    return eva_advise_node