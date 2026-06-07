import os
import shutil
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
# Paths relative to this file
KB_DIR = Path(__file__).parent / "kb"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize embedder and splitter once
_embeds = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

def _load_documents() -> List[Document]:
    """Load .txt and .pdf files from the KB directory.
    Supports easy extension to other formats.
    """
    docs: List[Document] = []
    if not KB_DIR.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {KB_DIR}")
    for file_path in KB_DIR.iterdir():
        if file_path.is_file():
            if file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path))
            elif file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                continue  # Skip unsupported formats
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = file_path.name
            docs.extend(loaded)
    return docs

def build_vector_store():
    """Create (or recreate) the Chroma DB from all KB files.
    Run this once after adding or updating KB content.
    """
    docs = _load_documents()
    chunks = _splitter.split_documents(docs)
    # Clean old DB if exists to avoid duplication
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    vectordb = Chroma.from_documents(chunks, _embeds, persist_directory=str(CHROMA_DIR))
    vectordb.persist()
    print(f"✅ Vector store built with {len(chunks)} chunks.")

def get_vector_store() -> Chroma:
    """Load the persisted Chroma DB, building it on‑the‑fly if missing."""
    if not CHROMA_DIR.exists():
        build_vector_store()
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=_embeds)

def retrieve_context(query: str, k: int = 4) -> List[Document]:
    """Return the top‑k most similar documents for a given query."""
    vectordb = get_vector_store()
    return vectordb.similarity_search(query, k=k)
