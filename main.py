import streamlit as st
from pathlib import Path
from src.data_loader import all_pdf
from src.search import RAGSearch
from src.vector_db import faissVectordB


@st.cache_resource
def init_pipeline(persist_dir: str = "faiss_store") -> RAGSearch:
    return RAGSearch(persist_dir)


def ask(query: str, top_k: int = 2, persist_dir: str = "faiss_store") -> str:
    pipeline = init_pipeline(persist_dir)
    return pipeline.search_and_summarize(query, top_k)


def build_from_directory(data_dir: str = "data", persist_dir: str = "faiss_store"):
    docs = all_pdf(data_dir)
    store = faissVectordB(persist_dir)
    store.build_from_docs(docs)
    store.save()
    st.cache_resource.clear()


def build_from_uploads(uploaded_files: list, persist_dir: str = "faiss_store"):
    save_dir = Path(persist_dir) / "uploads"
    save_dir.mkdir(parents=True, exist_ok=True)
    for f in uploaded_files:
        with open(save_dir / f.name, "wb") as out:
            out.write(f.getbuffer())
    build_from_directory(str(save_dir), persist_dir)
