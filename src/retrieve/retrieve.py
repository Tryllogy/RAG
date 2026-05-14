import pickle


class Retrieve:
    def __init__(
            self
    ):
        try:
            with open("data/processed/bm25_index/bm25.pkl", "rb") as f:
                self.retriever = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Index file not found. Please run the indexer first.")

    def retrieve(
            self,
            query: str,
            k: int = 5
    ):
        if self.retriever.k < k:
            print("k is larger than the default value, "
                  f"setting k to {self.retriever.k}")
        else:
            self.retriever.k = k
        results = self.retriever.invoke(query)
        return results
