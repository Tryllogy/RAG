from typing import Any
from ..chunker import Chunk
from langchain_community.retrievers import BM25Retriever
import pickle
import os


class Index:
    def __init__(self, llm: Any):
        self.llm: Any = llm
        self.retriver: BM25Retriever = None

    def index(
            self,
            max_chunk_size: int = 3000,
            k=5
    ) -> None:
        self.chunker = Chunk(max_chunk_size=max_chunk_size)
        self.chunker.chunk_files("data/raw/vllm-0.10.1")
        self.chunker.to_json()

        self.retriver = BM25Retriever.from_texts(
            texts=self.chunker.chunk_code + self.chunker.chunk_text,
            k=k
        )

        os.makedirs("data/processed/bm25_index", exist_ok=True)
        with open("data/processed/bm25_index/bm25.pkl", "wb") as f:
            pickle.dump(self.retriver, f)
