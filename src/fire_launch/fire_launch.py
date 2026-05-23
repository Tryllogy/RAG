from langchain_ollama import OllamaLLM
from ..evaluator import Evaluate
from ..index import Index
from ..answer import Answer
from ..search_dataset import SearchDataset
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
        answer_instance = Answer(question=question, k=k)
        result = answer_instance.answer()
        print(result)
        return self

    def search(self) -> "FireLaunch":
        pass

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 15,
            save_directory: str = "data/output/search_results"
    ) -> "FireLaunch":
        search_dataset: SearchDataset = SearchDataset(
            dataset_path, k, save_directory)
        search_dataset.search_dataset()
        return self

    def answer_dataset(self) -> "FireLaunch":
        pass

    def evaluate(
            self,
            student_answer_path: str,
            dataset_path: str,
            k: int = 15
    ) -> "FireLaunch":
        evaluate_instance: Evaluate = Evaluate(
            student_answer_path=student_answer_path,
            dataset_path=dataset_path
        )
        recall = evaluate_instance.evaluate(k=k)
        for k, v in recall.items():
            print(f"Recall@{k}: {v}")
        return self

    def __str__(self):
        return "Done"
