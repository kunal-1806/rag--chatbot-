import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingGenerator

class faissVectordB:
    def __init__(self,persist_dir: str,embedding_model_name: str = "all-MiniLM-L6-v2", index_file_path: str = "faiss_index.index",chunk_size: int = 512,chunk_overlap: int = 50):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index_file_path = index_file_path
        self.index = None
        self.id_to_text = {}
        self.metadata = []
        self.chunk_size = chunk_size  # Define the chunk size for splitting text
        self.chunk_overlap = chunk_overlap  # Define the chunk overlap for splitting text
        print(f"Initialized faissVectordB with embedding model: {embedding_model_name}, index file path: {index_file_path}, chunk size: {chunk_size}, and chunk overlap: {chunk_overlap}")
    def build_from_docs(self, embeddings:np.ndarray, ids: List[str], metadata: List[dict]):
        print(f"Building FAISS index from {len(ids)} documents.")
        emd_pipeline = EmbeddingGenerator(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emd_pipeline.chunk_doc(metadata)
        embeddings = emd_pipeline.generate_embeddings(chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.id_to_text = {i: text for i, text in enumerate(ids)}
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
    def add_embeddings(self,embeddings : np.ndarray,metadata: List[dict]) -> List[dict]:
        """
        Add embeddings to the FAISS index.

        Args:
            query (str): The query string to search for.
            top_k (int): The number of top results to return."""
        dim = embeddings.shape[1]

        if self.index is None:
             self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadata:
            self.metadata.extend(metadata)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)
