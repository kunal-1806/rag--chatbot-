from  typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",chunk_size: int = 500, chunk_overlap: int = 50):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_doc(self,documents: List[Any]) -> List[str]:
            """
            Chunk the documents into smaller pieces.

            Args:
                documents (List[Any]): The list of documents to be chunked.

            Returns:
                List[str]: A list of chunked text.
            """
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                # length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            print(f"[Debug] Chunking documents with chunk size {self.chunk_size} and overlap {self.chunk_overlap}.")
            chunks = text_splitter.split_documents(documents)
            return chunks
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray: 
        """
        Generate embeddings for the given chunks of text.

        Args:
            chunks (List[str]): The list of chunked text.

            """
        text = [chunk.page_content for chunk in chunks]
        print(f"[Debug] Chunk length: {len(text)}")
        print(f"[Debug] Generating embeddings for {len(text)} chunks.")
        embeddings = self.model.encode(text, convert_to_numpy=True)
        return embeddings
