from ..retrieve import Retrieve
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json


class Answer:
    def __init__(
            self,
            question: str = "",
            k: int = 5
    ) -> None:
        self.retriver: Retrieve = Retrieve()
        self.question: str = question
        self.k: int = k
        self.data_doc_path: str = "data/datasets/AnsweredQuestions/" \
            "dataset_code_public.json"
        self.data_code_path: str = "data/datasets/AnsweredQuestions/" \
            "dataset_docs_public.json"
        self.llm = OllamaLLM(model="qwen3:0.6b")

    def answer(
            self
    ) -> list:
        type_question: str = self.find_question(self.question)
        retrieved = self.retriver.retrieve(
            query=self.question, k=self.k, type_retriever=type_question)
        template = """Respond to the question using the provided context.
            Question: {question}
            Context: {context}"""

        chunks: list[str] = [c for c in retrieved]
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        result = chain.invoke(
            {"question": self.question, "context": chunks})
        return result

    def find_question(self, question: str):
        with open(self.data_code_path, "r") as f:
            content = json.load(f)
            content = content["rag_questions"]
        for c in content:
            if c["question"] == question:
                return "code"
        return "docs"
