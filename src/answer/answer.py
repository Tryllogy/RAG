from ..retrieve import Retrieve


class Answer:
    def __init__(
            self,
            question: str = "",
            k: int = 5
    ) -> None:
        self.retriver: Retrieve = Retrieve()
        self.question: str = question
        self.k: int = k

    def answer(
            self
    ) -> list:
        results = self.retriver.retrieve(query=self.question, k=self.k)
        return results
