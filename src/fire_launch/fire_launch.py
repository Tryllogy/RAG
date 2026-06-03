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

    @staticmethod
    def _validate_positive_int(value: int, param_name: str) -> int:
        """Validate that a parameter is a positive integer."""
        try:
            val = int(value)
            if val <= 0:
                raise ValueError(f"{param_name} must be positive (got {val})")
            return val
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid {param_name}: {e}")

    def index(
            self,
            max_chunk_size: int = 2000,
    ) -> "FireLaunch":
        max_chunk_size = self._validate_positive_int(
            max_chunk_size, "max_chunk_size")
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
        if not question.strip():
            raise ValueError("question cannot be empty")
        k = self._validate_positive_int(k, "k")
        answer_instance = Answer(question=question, k=k)
        answer_instance.answer()
        return self

    def search(
            self,
            dataset_path: str,
            query: str,
            save_directory: str = "data/output/search_results",
            k: int = 15
    ) -> "FireLaunch":
        search_dataset: SearchDataset = SearchDataset(
            dataset_path, k, save_directory)
        search_dataset.search_query(query)
        return self

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 15,
            save_directory: str = "data/output/search_results"
    ) -> "FireLaunch":
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        k = self._validate_positive_int(k, "k")
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
        responses: list = []
        with tqdm(total=len(student_results),
                  desc="Answering questions") as pbar:
            for r in student_results:
                sr = r["search_results"][0]
                answer_instance.question = sr.get("question", "")
                answer = answer_instance.answer()
                minimal = {
                    "question_id": sr.get("question_id"),
                    "question": sr.get("question"),
                    "retrieved_sources": [
                        {
                            "file_path": s.get("file_path"),
                            "first_character_index":
                            s.get("first_character_index"),
                            "last_character_index":
                            s.get("last_character_index")
                        }
                        for s in sr.get("retrieved_sources", [])
                    ],
                    "answer": answer
                }
                MinimalAnswer(**minimal)
                responses.append(minimal)
                pbar.update(1)
        k_val = student_results[0].get("k") if len(
            student_results) > 0 else None
        search_result: dict = {"search_results": responses, "k": k_val}
        StudentSearchResultsAndAnswer(**search_result)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        with open(os.path.join(
                save_directory, "search_results.json"), "w") as f:
            json.dump(search_result, f, indent=4)
        return self

    def evaluate(
            self,
            student_answer_path: str,
            dataset_path: str,
            k: int = 15
    ) -> "FireLaunch":
        if not os.path.exists(student_answer_path):
            raise FileNotFoundError(
                f"Student results not found: {student_answer_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        k = self._validate_positive_int(k, "k")
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
