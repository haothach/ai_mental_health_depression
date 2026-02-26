from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder


@dataclass(frozen=True)
class PineconeConfig:
    """Configuration for Pinecone vector database."""
    api_key: str
    index_name: str = "student-depression"
    cloud: str = "aws"
    region: str = "us-east-1"
    metric: str = "dotproduct"


def load_config_from_env(*, env_file: Optional[str] = None) -> PineconeConfig:
    """
    Load Pinecone configuration from environment variables.
    """
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing PINECONE_API_KEY. Add it to .env file or environment variables."
        )

    index_name = os.getenv("PINECONE_INDEX_NAME", "student-depression")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    metric = os.getenv("PINECONE_METRIC", "dotproduct")

    return PineconeConfig(
        api_key=api_key,
        index_name=index_name,
        cloud=cloud,
        region=region,
        metric=metric
    )


def create_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def _get_embedding_dimension(embeddings: HuggingFaceEmbeddings) -> int:
    """Get the dimension of the embedding model."""
    probe_vector = embeddings.embed_query("dimension probe")
    return len(probe_vector)


def ensure_pinecone_index(
    pc: Pinecone,
    *,
    index_name: str,
    dimension: int,
    metric: str,
    cloud: str,
    region: str,
) -> None:
    """
    Create Pinecone index if it doesn't exist.
    """
    if not pc.has_index(index_name):
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")


def fit_bm25(text_chunks: List[Document]) -> BM25Encoder:
    """
    Fit BM25 encoder on text chunks for sparse retrieval.
    """
    print("Fitting BM25 encoder...")
    bm25 = BM25Encoder().default()
    bm25.fit([doc.page_content for doc in text_chunks])
    print("BM25 encoder fitted successfully.")
    return bm25


def save_bm25(bm25: BM25Encoder, path: str | os.PathLike[str]) -> None:
    """
    Save BM25 encoder to disk for reuse during query phase.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 encoder saved to {path}")


def load_bm25(path: str | os.PathLike[str]) -> BM25Encoder:
    """
    Load BM25 encoder from disk.
    """
    with open(path, "rb") as f:
        bm25 = pickle.load(f)
    print(f"BM25 encoder loaded from {path}")
    return bm25


def build_hybrid_retriever_and_upsert(
    *,
    text_chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    config: PineconeConfig,
    bm25_save_path: Optional[str | os.PathLike[str]] = None,
    top_k: int = 5,
) -> PineconeHybridSearchRetriever:
    """
    Build hybrid retriever and upsert documents to Pinecone.
    """
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=config.api_key)

    dim = _get_embedding_dimension(embeddings)
    print(f"Embedding dimension: {dim}")

    # Ensure index exists
    ensure_pinecone_index(
        pc,
        index_name=config.index_name,
        dimension=dim,
        metric=config.metric,
        cloud=config.cloud,
        region=config.region,
    )

    # Get index
    index = pc.Index(config.index_name)

    # Fit BM25
    bm25 = fit_bm25(text_chunks)
    
    # Save BM25 if path provided
    if bm25_save_path:
        save_bm25(bm25, bm25_save_path)

    # Create hybrid retriever
    print("Creating hybrid retriever...")
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=index,
        top_k=top_k,
    )

    # Upsert documents
    print(f"Upserting {len(text_chunks)} chunks to Pinecone...")
    texts = [d.page_content for d in text_chunks]
    metadatas = [d.metadata for d in text_chunks]
    retriever.add_texts(texts=texts, metadatas=metadatas)
    print("Upsert completed successfully.")

    return retriever