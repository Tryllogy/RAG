from typing import Any
from ..chunker import Chunk
from langchain_community.retrievers import BM25Retriever


class Index:
    def __init__(self, llm: Any):
        self.llm: Any = llm
        self.retriver: BM25Retriever = None

    def index(
            self,
            max_chunk_size: int = 3000,
            k=5
    ) -> BM25Retriever:
        self.chunker = Chunk(max_chunk_size=max_chunk_size)
        self.chunker.chunk_files("datasets_public/vllm-0.10.1")
        self.chunker.to_json()

        self.retriver = BM25Retriever.from_texts(
            texts=self.chunker.chunk_code + self.chunker.chunk_text,
            k=k
        )

        return self.retriver
