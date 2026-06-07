from typing import Any, Callable
from tqdm import tqdm
from ..chunker import Chunk
from langchain_community.retrievers import BM25Retriever
import os
import numpy as np
from ..process_code import ProcessCode
import time


class Index:
    def __init__(self, llm: Any, max_chunk_size: int = 2000, k: int = 15):
        """Initialize the indexer with configuration.

        Args:
            llm: The language model instance for processing.
            max_chunk_size: Maximum chunk size in characters. Defaults to
                2000.
            k: Default number of results for retrieval. Defaults to 15.
        """
        self.llm: Any = llm
        self.max_chunk_size: int = max_chunk_size
        self.k: int = k
        self.retriver: BM25Retriever | None = None
        self.done: bool = False

    def index(self) -> None:
        """Create chunked documents and BM25 indices.

        Processes documents, chunks them appropriately (code vs text),
        creates BM25 indices for efficient retrieval, and saves indices
        to disk.
        """
        start_time = time.time()
        self.chunker = Chunk(max_chunk_size=self.max_chunk_size)
        self.chunker.get_files("data/raw/vllm-0.10.1")
        self.chunker.chunk_files()
        self.chunker.to_json()

        code_path = "data/processed/bm25_index/bm25_code_index.npy"
        text_path = "data/processed/bm25_index/bm25_docs_index.npy"

        if not os.path.exists("data/processed/bm25_index"):
            os.makedirs("data/processed/bm25_index", exist_ok=True)

        chunk_code = self.chunker.chunk_code_with_source
        texts = [chunk["chunk"] for chunk in chunk_code]
        metadatas = [{"file_path": chunk["file_path"],
                      "first_character_index": chunk["first_character_index"],
                      "last_character_index": chunk["last_character_index"],
                      "name": chunk["name"],
                      "type": chunk["type"],
                      "docstring": chunk["docstring"],
                      "parent": chunk["parent"]}
                     for chunk in chunk_code]
        self.save_index(
            code_path,
            texts,
            metadatas,
            preprocess_func=ProcessCode._preprocess_code,
        )

        chunk_text = self.chunker.chunk_text_with_source
        texts = [chunk["chunk"] for chunk in chunk_text]
        metadatas = [{"file_path": chunk["file_path"],
                      "first_character_index": chunk["first_character_index"],
                      "last_character_index": chunk["last_character_index"],
                      } for chunk in chunk_text]
        self.save_index(
            text_path,
            texts,
            metadatas,
            preprocess_func=ProcessCode._preprocess_text,
        )

        elapsed = time.time() - start_time
        print(f"Indexing complete in {elapsed:.2f}s")

    def save_index(
            self,
            file_path: str,
            texts: list[str],
            metadatas: list[dict],
            preprocess_func: Callable
    ) -> None:
        """Create and save a BM25 index.

        Builds a BM25 retriever from texts and metadata, then serializes
        it to disk.

        Args:
            file_path: Path where the index will be saved.
            texts: List of document texts to index.
            metadatas: List of metadata dictionaries corresponding to texts.
            preprocess_func: Optional preprocessing function for text
                tokenization. Defaults to None.
        """
        with tqdm(total=1, desc="Indexing chunks") as pbar:
            self.retriver = BM25Retriever.from_texts(
                texts=texts,
                metadatas=metadatas,
                k=self.k,
                preprocess_func=preprocess_func,
            )
            pbar.update(1)
        with open(file_path, "wb") as f:
            with tqdm(total=1, desc="Saving BM25 for docs index") as pbar:
                data: np.ndarray[Any, Any] = np.array([{
                    "texts": texts,
                    "metadatas": metadatas,
                    "k": self.k,
                }], dtype=object)
                np.save(
                    f,
                    data,
                    allow_pickle=True,
                )
                pbar.update(1)
