from typing import Any
from tqdm import tqdm
from ..chunker import Chunk
from langchain_community.retrievers import BM25Retriever
import pickle
import os


class Index:
    def __init__(self, llm: Any, max_chunk_size: int = 3000, k: int = 15):
        self.llm: Any = llm
        self.max_chunk_size: int = max_chunk_size
        self.k: int = k
        self.retriver: BM25Retriever = None
        self.done: bool = False

    def index(
            self
    ) -> None:
        self.chunker = Chunk(max_chunk_size=self.max_chunk_size)
        self.chunker.get_files("data/raw/vllm-0.10.1")
        self.chunker.chunk_files()
        self.chunker.to_json()

        with tqdm(total=1, desc="Indexing chunks") as pbar:
            self.retriver = BM25Retriever.from_texts(
                texts=self.chunker.chunk_code + self.chunker.chunk_text,
                k=self.k
            )
            pbar.update(1)

        os.makedirs("data/processed/bm25_index", exist_ok=True)
        with open("data/processed/bm25_index/bm25.pkl", "wb") as f:
            with tqdm(total=1, desc="Saving BM25 index") as pbar:
                pickle.dump(self.retriver, f)
                pbar.update(1)
