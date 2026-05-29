import numpy as np
from langchain_community.retrievers import BM25Retriever
from ..process_code import ProcessCode


class Retrieve:
    def __init__(
            self
    ):
        self.retriever: dict = {
            "code": None,
            "docs": None
        }
        try:
            with open("data/processed/bm25_index/bm25_code_index.npy",
                      "rb") as f:
                loaded = np.load(f, allow_pickle=True)
                payload = loaded.item() if loaded.size == 1 else loaded
                self.retriever["code"] = self._build_retriever(
                    payload,
                    ProcessCode._preprocess_code,
                )
            with open("data/processed/bm25_index/bm25_docs_index.npy",
                      "rb") as f:
                loaded = np.load(f, allow_pickle=True)
                payload = loaded.item() if loaded.size == 1 else loaded
                self.retriever["docs"] = self._build_retriever(
                    payload,
                    ProcessCode._preprocess_text,
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Index file not found. Please run the indexer first.")

    def _build_retriever(self, payload, preprocess_func):
        if isinstance(payload, BM25Retriever):
            return payload
        if not isinstance(payload, dict):
            raise TypeError("Invalid BM25 index payload")
        return BM25Retriever.from_texts(
            texts=payload["texts"],
            metadatas=payload["metadatas"],
            k=payload.get("k", 15),
            preprocess_func=preprocess_func,
        )

    def retrieve(
            self,
            query: str,
            k: int = 15,
            type_retriever: str = "code"
    ):
        save_k: int = self.retriever[type_retriever].k
        if k > save_k:
            print(
                f"k ({k}) is larger than the stored default ({save_k}), "
                f"using {k}")
        self.retriever[type_retriever].k = k
        results = self.retriever[type_retriever].invoke(query)
        self.retriever[type_retriever].k = save_k
        return results
