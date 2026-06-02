from ..retrieve import Retrieve
from langchain_ollama import ChatOllama
import json


class Answer:
    def __init__(
            self,
            question: str = "",
            k: int = 10
    ) -> None:
        self.retriver: Retrieve = Retrieve()
        self.question: str = question
        self.k: int = k
        self.data_doc_path: str = "data/datasets/AnsweredQuestions/" \
            "dataset_code_public.json"
        self.data_code_path: str = "data/datasets/AnsweredQuestions/" \
            "dataset_docs_public.json"
        self.llm = ChatOllama(model="qwen3:0.6b")
        self.prompt_message: str = "You are a helpful assistant that " \
            "answers questions based ONLY on the provided context. " \
            "If the answer is not in the context, say 'I don't have " \
            "this information in the provided context'. " \
            "Make your answer concise and directly address the question. " \
            "Respond in English."

    def answer(
            self
    ) -> list:
        type_question: str = self.find_question(self.question)
        retrieved = self.retriver.retrieve(
            query=self.question, k=self.k, type_retriever=type_question)

        chunks: list[str] = [c.page_content for c in retrieved]
        message: list = [
            (
                "system",
                self.prompt_message
            ),
            (
                "user",
                f"Question: {self.question}\n\nContext:\n"
                f"{chr(10).join(chunks)}\n\nAnswer:"
            )
        ]
        result = self.llm.invoke(message)
        return result.content.split("</think>")

    def find_question(self, question: str):
        with open(self.data_code_path, "r") as f:
            content = json.load(f)
            content = content["rag_questions"]
        for c in content:
            if c["question"] == question:
                return "code"
        return "docs"
