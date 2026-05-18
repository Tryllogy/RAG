import pickle


class Retrieve:
    def __init__(
            self
    ):
        self.retriever: dict = {
            "code": None,
            "text": None
        }
        try:
            with open("data/processed/bm25_index/bm25_code_index.pkl",
                      "rb") as f:
                self.retriever["code"] = pickle.load(f)
            with open("data/processed/bm25_index/bm25_text_index.pkl",
                      "rb") as f:
                self.retriever["text"] = pickle.load(f)
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
