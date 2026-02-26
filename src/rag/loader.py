from __future__ import annotations

import os
from typing import Iterable, List, Optional, Union

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

def load_pdf_files(
    data_dir: Union[str, os.PathLike[str]],
    glob: str = "*.pdf",
    silent_errors: bool = True,
) -> List[Document]:
    """
    Load all PDFs from a directory into LangChain Documents.
    """
    loader = DirectoryLoader(
        str(data_dir),
        glob=glob,
        loader_cls=PyPDFLoader, # type: ignore
        silent_errors=silent_errors,
    )
    return loader.load()


def filter_to_minimal_docs(
    docs: Iterable[Document],
) -> List[Document]:
    """
    Keep only minimal, stable metadata keys (helps reduce payload size in vector DB).
    """
    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source", "")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs