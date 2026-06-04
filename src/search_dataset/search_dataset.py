import json
import os

from tqdm import tqdm
from ..retrieve import Retrieve
from ..validators import StudentSearchResults


class SearchDataset():
    def __init__(
            self,
            dataset_path: str,
            k: int = 15,
            save_dir: str = "data/output/search_results"
    ) -> None:
        """Initialize dataset searcher.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of documents to retrieve per query. Defaults to 15.
            save_dir: Directory to save results. Defaults to
                "data/output/search_results".
        """
        self.dataset_path: str = dataset_path
        self.k: int = k
        self.save_dir: str = save_dir
        self.retriever = Retrieve()

    def search_dataset(self) -> None:
        """Search for all questions in the dataset.

        Iterates through all questions in the dataset, retrieves relevant
        documents for each, and saves results.
        """
        search_results_list: list = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data: dict = json.load(f)
            data = data["rag_questions"]
        type_json: str = self.dataset_path.split("_")[1]
        with tqdm(data, desc="Searching dataset") as pbar:
            for question in data:
                q: str = question["question"]
                id_q: str = question["question_id"]
                retrieved_data = self.retriever.retrieve(
                    question["question"], self.k, type_json)
                metadatas: list = [r.metadata for r in retrieved_data]
                search_result = {
                    "question_id": id_q,
                    "question": q,
                    "question_str": q,
                    "retrieved_sources": metadatas
                }
                search_results_list.append(search_result)
                pbar.update(1)
        final_result = {
            "search_results": search_results_list,
            "k": self.k
        }
        StudentSearchResults(**final_result)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir,
                               f"dataset_{type_json}_public.json"), "w",
                  encoding="utf-8") as f:
            json.dump(final_result, f, indent=4)

    def search_query(self, query: str) -> None:
        """Search for a specific query in the dataset.

        Finds and searches for a specific question matching the query
        string.

        Args:
            query: The query string to match and search.
        """
        search_results_list: list = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data: dict = json.load(f)
            data = data["rag_questions"]
        type_json: str = self.dataset_path.split("_")[1]

        for question in data:
            if question["question"] != query:
                continue
            q: str = question["question"]
            id_q: str = question["question_id"]
            retrieved_data = self.retriever.retrieve(
                question["question"], self.k, type_json)
            metadatas: list = [r.metadata for r in retrieved_data]
            search_result = {
                "question_id": id_q,
                "question": q,
                "question_str": q,
                "retrieved_sources": metadatas
            }
            search_results_list.append(search_result)
        final_result = {
            "search_results": search_results_list,
            "k": self.k
        }
        StudentSearchResults(**final_result)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir,
                               f"dataset_{type_json}_public.json"), "w",
                  encoding="utf-8") as f:
            json.dump(final_result, f, indent=4)
