from __future__ import annotations

import json
from uuid import uuid4
from typing import Any, Dict, List, Literal, TypedDict

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from graph.build_graph import build_app


Role = Literal["user", "assistant"]


class ChatTurn(TypedDict):
    role: Role
    content: str


def _to_turns(messages: List[BaseMessage]) -> List[ChatTurn]:
    turns: List[ChatTurn] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            turns.append({"role": "user", "content": m.content}) # type: ignore
        elif isinstance(m, AIMessage):
            turns.append({"role": "assistant", "content": m.content}) # type: ignore
        else:
            # Ignore other message types (SystemMessage, ToolMessage, etc.) in UI
            pass
    return turns


def _turns_to_lc_messages(turns: List[ChatTurn]) -> List[BaseMessage]:
    lc: List[BaseMessage] = []
    for t in turns:
        if t["role"] == "user":
            lc.append(HumanMessage(content=t["content"]))
        else:
            lc.append(AIMessage(content=t["content"]))
    return lc


@st.cache_resource
def _get_app():
    # Compiled LangGraph app (with MemorySaver) from your build_graph.py
    return build_app()


def _init_session_state():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    if "turns" not in st.session_state:
        st.session_state.turns = []  # List[ChatTurn]
    if "last_state" not in st.session_state:
        st.session_state.last_state = None  # last returned state dict


def _reset_chat():
    st.session_state.thread_id = str(uuid4())
    st.session_state.turns = []
    st.session_state.last_state = None


def main():
    st.set_page_config(page_title="Student Depression Graph Demo", layout="wide")
    _init_session_state()

    st.title("LangGraph Demo (Student Wellbeing)")
    st.caption("Streamlit UI to run your StateGraph with memory (thread_id-based).")

    with st.sidebar:
        st.subheader("Session")
        st.code(st.session_state.thread_id, language="text")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset chat", use_container_width=True):
                _reset_chat()
                st.rerun()
        with col2:
            if st.button("Rebuild app", use_container_width=True):
                # Clear cached app (forces rebuild next run)
                _get_app.clear()
                st.rerun()

        st.divider()
        st.subheader("Latest state (debug)")
        state = st.session_state.last_state
        if isinstance(state, dict):
            st.markdown("**profile**")
            st.json(state.get("profile", {}))
            st.markdown("**prediction**")
            st.json(state.get("prediction", {}))
            st.markdown("**missing_fields**")
            st.json(state.get("missing_fields", []))
            st.markdown("**intent / pending / completed**")
            st.json(
                {
                    "intent": state.get("intent"),
                    "pending_agents": state.get("pending_agents", []),
                    "completed_agents": state.get("completed_agents", []),
                }
            )
        else:
            st.info("No state yet. Send a message to start.")

    # Render chat history
    for t in st.session_state.turns:
        with st.chat_message(t["role"]):
            st.markdown(t["content"])

    user_text = st.chat_input("Type your message…")
    if not user_text:
        return

    # Add user turn
    st.session_state.turns.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    app = _get_app()

    # Build LC messages for the graph input
    lc_messages = _turns_to_lc_messages(st.session_state.turns)

    # Invoke graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result_state: Dict[str, Any] = app.invoke(
                    {"messages": lc_messages},
                    {"configurable": {"thread_id": st.session_state.thread_id}},
                )
            except Exception as e:
                st.error(f"Graph error: {e}")
                return

        st.session_state.last_state = result_state

        # Append any new assistant messages produced by the graph
        out_messages = result_state.get("messages", [])
        # Only take the messages that were added after our input list length
        new_msgs = out_messages[len(lc_messages) :] if isinstance(out_messages, list) else []

        new_turns = _to_turns(new_msgs)

        # Only show the latest assistant message (instead of all new messages)
        assistant_turns = [t for t in new_turns if t["role"] == "assistant"]
        if not assistant_turns:
            # Fallback: show whole state if no assistant message returned
            st.markdown("No assistant message returned. Raw state:")
            st.code(json.dumps(result_state, ensure_ascii=False, indent=2))
            return

        last_assistant_turn = assistant_turns[-1]
        st.markdown(last_assistant_turn["content"])
        st.session_state.turns.append(last_assistant_turn)


if __name__ == "__main__":
    main()