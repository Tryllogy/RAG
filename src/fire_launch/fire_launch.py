import json

from langchain_ollama import OllamaLLM
from ..index import Index
from ..answer import Answer
import os


class FireLaunch:
    def __init__(self):
        self.llm: OllamaLLM = OllamaLLM(model="qwen3")

    def index(
            self,
            max_chunk_size: int = 3000,
    ) -> "FireLaunch":
        if os.path.exists("data/processed/chunks") and \
                os.path.exists("data/processed/bm25_index"):
            print("Files already indexed")
            return self
        index_instance = Index(llm=self.llm,
                               max_chunk_size=max_chunk_size)
        index_instance.index()
        return self

    def answer(
            self,
            question: str = "",
            k: int = 5
    ) -> "FireLaunch":
        recall: dict = {
            1: 0,
            3: 0,
            5: 0,
            10: 0
        }
        with open("data/datasets_public/AnsweredQuestions/dataset_code_public.json", "r") as f:
            data = json.load(f)
            for item in data["rag_questions"]:
                answer_instance = Answer(question=item["question"], k=k)
                result = answer_instance.answer()
                for r in recall:
                    recall[r] += answer_instance.retriver.recall_score[r]
        for r in recall.items():
            print(f"Recall@{r[0]}: ", round(r[1] / (len(data["rag_questions"]) * 100), 3))
        return self

    def search(self) -> "FireLaunch":
        pass

    def search_dataset(self) -> "FireLaunch":
        pass

    def answer_dataset(self) -> "FireLaunch":
        pass

    def evaluate(self) -> "FireLaunch":
        pass

    def __str__(self):
        return "Done"
