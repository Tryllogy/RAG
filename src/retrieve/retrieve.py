import numpy as np


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
                self.retriever["code"] = (
                    loaded.item() if loaded.size == 1 else loaded)
            with open("data/processed/bm25_index/bm25_docs_index.npy",
                      "rb") as f:
                loaded = np.load(f, allow_pickle=True)
                self.retriever["docs"] = (
                    loaded.item() if loaded.size == 1 else loaded)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Index file not found. Please run the indexer first.")

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
