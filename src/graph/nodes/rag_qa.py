from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.types import Command

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone

from state_types.state import ChatState
from utils.utils import get_last_user_message, load_prompt, mark_agent_completed
from rag.retriever import load_bm25, load_config_from_env, create_embeddings


def build_hybrid_retriever_from_env(
    *,
    bm25_path: Path,
    top_k: int = 3,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> PineconeHybridSearchRetriever:
    config = load_config_from_env()
    embeddings = create_embeddings(embed_model_name)
    bm25 = load_bm25(bm25_path)

    pc = Pinecone(api_key=config.api_key)
    index = pc.Index(config.index_name)

    return PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=index,
        top_k=top_k,
    )


def _to_text(x: Any) -> str:
    """Best-effort normalize chain outputs to a string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    
    if isinstance(x, dict):
        return (
            x.get("answer")
            or x.get("text")
            or x.get("output_text")
            or x.get("result")
            or str(x)
        )
    # LangChain message-like objects
    content = getattr(x, "content", None)
    if isinstance(content, str):
        return content
    return str(x)


def make_rag_retriever(retriever: Any) -> Callable[[str], List[Any]]:
    def rag_retriever(query: str) -> List[Any]:
        return retriever.invoke(query)
    return rag_retriever


def make_rag_qa_node(
    *,
    llm: Any,
    prompt_dir: Path,
    retriever: Any,
    prompt_name: str = "14_rag_qa_agent.md",
    store_docs_in_state: bool = True,
) -> Callable[[ChatState], Command]:

    def rag_qa_node(state: ChatState) -> Command:
        messages = state.get("messages", []) or []
        user_text = get_last_user_message(messages)

        retrieved_docs = retriever.invoke(user_text)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt(prompt_name, prompts_dir=str(prompt_dir))),
                (
                    "human",
                    "QUESTION:\n{user_query}\n\n"
                    "CONTEXT (retrieved docs):\n{context}\n\n"
                    "Answer the question using only the provided context when possible.",
                ),
            ]
        )

        stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
        answer = stuff_chain.invoke(
            {
                "user_query": user_text,
                "context": retrieved_docs,
            }
        )
        answer_text = _to_text(answer)

        update: Dict[str, Any] = {
            **state,
            "messages": messages + [AIMessage(content=answer_text)],
            **mark_agent_completed(state, "rag_responder")
        }
        if store_docs_in_state:
            update["rag_query"] = user_text
            update["rag_docs"] = retrieved_docs

        return Command(update=update, goto=END)

    return rag_qa_node