from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from state_types.state import ChatState, IntakerOutput

from graph.nodes.supervisor import make_supervisor_node
from graph.nodes.intaker import make_intaker_node
from graph.nodes.waiting_for_user import make_waiting_for_user_node
from graph.nodes.direct_response import make_direct_response_node
from graph.nodes.predict import make_predict_node
from graph.nodes.eva_advise import make_eva_advise_node
from graph.nodes.rag_qa import build_hybrid_retriever_from_env, make_rag_qa_node

from predict.helper import load_model_and_scaler
from predict.predict import predict_from_profile


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    src_dir: Path
    prompt_dir: Path
    model_predict_dir: Path
    bm25_path: Path


def get_default_paths() -> AppPaths:
    
    src_dir = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[2]

    prompt_dir = src_dir / "prompts"
    model_predict_dir = project_root / "model" / "predict"
    bm25_path = project_root / "model" / "rag" / "artifacts" / "bm25_encoder.pkl"

    return AppPaths(
        project_root=project_root,
        src_dir=src_dir,
        prompt_dir=prompt_dir,
        model_predict_dir=model_predict_dir,
        bm25_path=bm25_path,
    )


def build_llms(
    *,
    model_chat: str = "gpt-4o-mini",
    model_super: str = "gpt-4o",
    temperature: float = 0.0,
) -> Tuple[ChatOpenAI, ChatOpenAI]:
    llm = ChatOpenAI(model=model_chat, temperature=temperature)
    llm_super = ChatOpenAI(model=model_super, temperature=temperature)
    return llm, llm_super


def build_graph(
    *,
    prompt_dir: Optional[Path] = None,
    bm25_path: Optional[Path] = None,
    model_predict_dir: Optional[Path] = None,
    rag_top_k: int = 3,
    rag_embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    advisor_prompt_name: str = "15_evaluate_and_advise_agent.md",
    rag_qa_prompt_name: str = "14_rag_qa_agent.md",
    predictor_completed_name: str = "predictor",
    advisor_completed_name: str = "advisor_agent",
) -> StateGraph:

    load_dotenv(override=True)

    paths = get_default_paths()
    prompt_dir = prompt_dir or paths.prompt_dir
    bm25_path = bm25_path or paths.bm25_path
    model_predict_dir = model_predict_dir or paths.model_predict_dir

    # LLMs (same as notebook)
    llm, llm_super = build_llms()

    # Intaker structured output (same as notebook)
    intaker_llm = llm.with_structured_output(IntakerOutput)

    # Predictor artifacts (same as notebook)
    predict_model, predict_scaler = load_model_and_scaler(str(model_predict_dir))

    # RAG retriever (same retriever type as notebook)
    hybrid_retriever = build_hybrid_retriever_from_env(
        bm25_path=bm25_path,
        top_k=rag_top_k,
        embed_model_name=rag_embed_model_name,
    )

    # Build node callables (dependency-injected)
    supervisor_node = make_supervisor_node(llm_super=llm_super, prompt_dir=prompt_dir)
    intaker_node = make_intaker_node(intaker_llm=intaker_llm, prompt_dir=prompt_dir)
    waiting_for_user_node = make_waiting_for_user_node(llm=llm, prompt_dir=prompt_dir)
    direct_responder_node = make_direct_response_node(llm=llm, prompt_dir=prompt_dir)

    predict_node = make_predict_node(
        model=predict_model,
        scaler=predict_scaler,
        completed_agent_name=predictor_completed_name,
    )

    # rag_retriever signature expected by eva_advise: (query: str) -> list[Document]
    rag_retriever = lambda q: hybrid_retriever.invoke(q)

    advisor_node = make_eva_advise_node(
        llm=llm,
        prompt_dir=prompt_dir,
        rag_retriever=rag_retriever,
        prompt_name=advisor_prompt_name,
        completed_agent_name=advisor_completed_name,
    )

    rag_responder_node = make_rag_qa_node(
        llm=llm,
        prompt_dir=prompt_dir,
        retriever=hybrid_retriever,
        prompt_name=rag_qa_prompt_name,
    )

    # Graph
    graph = StateGraph(ChatState)

    graph.add_node("supervisor", supervisor_node) # type: ignore
    graph.add_node("intaker", intaker_node) # type: ignore
    graph.add_node("waiting_for_user", waiting_for_user_node) # type: ignore
    graph.add_node("direct_responder", direct_responder_node) # type: ignore
    graph.add_node("advisor", advisor_node) # type: ignore
    graph.add_node("rag_responder", rag_responder_node) # type: ignore
    graph.add_node("predictor", predict_node) # type: ignore
 
    graph.set_entry_point("supervisor")
    return graph


def build_app(**kwargs: Any):

    graph = build_graph(**kwargs)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)