import json
import os
import uuid

from tqdm import tqdm
from ..retrieve import Retrieve
from ..validators import StudentSearchResults


class SearchDataset():
    def __init__(
            self,
            k: int = 15,
            save_dir: str = "data/output/search_results",
            dataset_path: str = "",
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
                    "retrieved_sources": metadatas
                }
                search_results_list.append(search_result)
                pbar.update(1)
        final_result = {
            "search_results": search_results_list,
            "k": self.k
        }
        StudentSearchResults.model_validate(final_result)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir,
                               f"dataset_{type_json}_public.json"), "w",
                  encoding="utf-8") as f:
            json.dump(final_result, f, indent=4)

    def search_query(self, query: str) -> None:
        """Search for a single query and print results.

        Args:
            query: The search query string.
        """
        retrieved_data = self.retriever.retrieve(query, self.k, "code")
        metadatas: list = [r.metadata for r in retrieved_data]
        uuid_str: str = str(uuid.uuid4())
        search_result = [{
            "question_id": uuid_str,
            "question": query,
            "retrieved_sources": metadatas
        }]
        data: dict = {"search_results": search_result, "k": self.k}
        StudentSearchResults.model_validate(data)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(
            self.save_dir, "query_search_results.json"), "w",
                  encoding="utf-8") as f:
            json.dump(data, f, indent=4)
