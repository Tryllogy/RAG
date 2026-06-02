from langchain_ollama import OllamaLLM
from ..evaluator import Evaluate
from ..index import Index
from ..answer import Answer
from ..search_dataset import SearchDataset
from ..validators import StudentSearchResultsAndAnswer, MinimalAnswer
from tqdm import tqdm
import json
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
        answer_instance.answer()
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

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str
    ) -> "FireLaunch":
        answer_instance = Answer()
        with open(student_search_results_path, "r") as f:
            student_results = json.load(f)
            student_results = student_results["search_results"]
        responses: list = []
        with tqdm(total=len(student_results),
                  desc="Answering questions") as pbar:
            for r in student_results:
                answer_instance.question = r["question"]
                answer = answer_instance.answer()
                response = {
                    "search_results": r["search_results"]
                }
                response["search_results"]["answer"] = answer
                MinimalAnswer(**response)
                responses.append(response)
                pbar.update(1)
        search_result: dict = {"search_results": responses}
        StudentSearchResultsAndAnswer(**search_result)
        with open(save_directory, "w") as f:
            json.dump(search_result, f)
        return self

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
