from dotenv import load_dotenv
from src.vector_db import faissVectordB
import os
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter

load_dotenv()  # Load environment variables from .env file

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "nvidia/nemotron-3-nano-30b-a3b:free",top_k: int = 2):
        self.top_k = top_k
        self.vectorstore = faissVectordB(persist_dir, embedding_model)
        # Load existing index if present
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.vectorstore.load()
        openrouter_api_key = os.getenv("open_router_key")
        self.llm = ChatOpenRouter(
            openrouter_api_key=openrouter_api_key,
            model=llm_model,
            temperature=0,
            max_tokens=1024,
            max_retries=2,
        )
        print(f"[INFO] LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 2) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content 

