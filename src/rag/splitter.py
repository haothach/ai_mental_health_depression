from __future__ import annotations

from typing import Iterable, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    docs: Iterable[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 80,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """
    Split Documents into overlapping chunks for retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )
    return splitter.split_documents(list(docs))