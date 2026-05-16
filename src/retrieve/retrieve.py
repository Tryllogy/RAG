import pickle
from ..evaluator import Evaluate


class Retrieve:
    recall_score: dict = {
        1: 0,
        3: 0,
        5: 0,
        10: 0
    }

    def __init__(
            self
    ):
        try:
            with open("data/processed/bm25_index/bm25.pkl", "rb") as f:
                self.retriever = pickle.load(f)
            self.evaluator = Evaluate()
        except FileNotFoundError:
            raise FileNotFoundError(
                "Index file not found. Please run the indexer first.")

    def retrieve(
            self,
            query: str,
            k: int = 5
    ):
        to_retrieve: list = [1, 3, 5, 10]
        if self.retriever.k < k:
            print("k is larger than the default value, "
                  f"setting k to {self.retriever.k}")
        else:
            self.retriever.k = k
        results = self.retriever.invoke(query)
        self.evaluator.get_source_question(query)
        for r in to_retrieve:
            self.recall_score[r] += self.evaluator.evaluate(results, r)
        return results
