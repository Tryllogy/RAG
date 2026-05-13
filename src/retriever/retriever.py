from langchain_community import BM25Retriever


class Retrieve:
    def __init__(
            self,
            retriever: BM25Retriever
    ):
        self.retriver: BM25Retriever = retriever

    def retrieve(self, query: str):

        results = self.retriver.invoke(query)
        return results
