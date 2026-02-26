from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.loader import load_pdf_files, filter_to_minimal_docs
from src.rag.splitter import split_documents
from src.rag.retriever import (
    load_config_from_env,
    create_embeddings,
    build_hybrid_retriever_and_upsert,
)


def main():
    """Main function to build RAG index."""
    print("=" * 60)
    print("Building RAG Index for Student Depression Chatbot")
    print("=" * 60)

    # Configuration
    DATA_DIR = project_root / "data" / "rag"
    BM25_SAVE_PATH = project_root / "model" / "rag" / "artifacts" / "bm25_encoder.pkl"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 80
    TOP_K = 5

    print(f"\nConfiguration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  BM25 save path: {BM25_SAVE_PATH}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  Chunk overlap: {CHUNK_OVERLAP}")
    print(f"  Top-k: {TOP_K}")

    # Step 1: Load PDFs
    print("\n" + "=" * 60)
    print("Step 1: Loading PDF files")
    print("=" * 60)
    extracted_data = load_pdf_files(DATA_DIR)
    print(f"Loaded {len(extracted_data)} pages from PDFs")

    # Step 2: Filter metadata
    print("\n" + "=" * 60)
    print("Step 2: Filtering metadata")
    print("=" * 60)
    minimal_docs = filter_to_minimal_docs(extracted_data)
    print(f"Filtered to {len(minimal_docs)} documents with minimal metadata")

    # Step 3: Split into chunks
    print("\n" + "=" * 60)
    print("Step 3: Splitting documents into chunks")
    print("=" * 60)
    text_chunks = split_documents(
        minimal_docs,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    print(f"Split into {len(text_chunks)} chunks")

    # Step 4: Create embeddings
    print("\n" + "=" * 60)
    print("Step 4: Loading embedding model")
    print("=" * 60)
    embeddings = create_embeddings(model_name=EMBEDDING_MODEL)
    print(f"Embedding model loaded: {EMBEDDING_MODEL}")

    # Step 5: Load Pinecone config
    print("\n" + "=" * 60)
    print("Step 5: Loading Pinecone configuration")
    print("=" * 60)
    config = load_config_from_env()
    print(f"Pinecone index: {config.index_name}")
    print(f"Cloud: {config.cloud}")
    print(f"Region: {config.region}")

    # Step 6: Build retriever and upsert
    print("\n" + "=" * 60)
    print("Step 6: Building retriever and upserting to Pinecone")
    print("=" * 60)
    retriever = build_hybrid_retriever_and_upsert(
        text_chunks=text_chunks,
        embeddings=embeddings,
        config=config,
        bm25_save_path=BM25_SAVE_PATH,
        top_k=TOP_K,
    )

    # Test query
    print("\n" + "=" * 60)
    print("Step 7: Testing retriever")
    print("=" * 60)
    test_query = "What is depression?"
    print(f"Test query: '{test_query}'")
    results = retriever.invoke(test_query)
    print(f"Retrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        score = doc.metadata.get("score", "N/A")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"\n  {i}. Source: {source}")
        print(f"     Score: {score}")
        print(f"     Preview: {preview}...")

    print("\n" + "=" * 60)
    print("✓ Index built successfully!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. BM25 encoder saved to: {BM25_SAVE_PATH}")
    print(f"  2. {len(text_chunks)} chunks indexed in Pinecone")
    print(f"  3. Ready to use in query.py or chatbot")


if __name__ == "__main__":
    main()