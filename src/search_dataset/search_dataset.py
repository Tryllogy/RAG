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
        self.dataset_path: str = dataset_path
        self.k: int = k
        self.save_dir: str = save_dir
        self.retriever = Retrieve()

    def search_dataset(self):
        retrieved: list = []
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
                    "search_results": [{
                        "question_id": id_q,
                        "question": q,
                        "retrieved_sources": metadatas
                    }],
                    "k": self.k
                }
                StudentSearchResults(**search_result)
                retrieved.append(search_result)
                pbar.update(1)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir,
                               f"dataset_{type_json}_public.json"), "w",
                  encoding="utf-8") as f:
            json.dump(retrieved, f, indent=4)

    def search_query(
            self,
            query: str,
    ):
        retrieved: list = []
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
                "search_results": [{
                    "question_id": id_q,
                    "question": q,
                    "retrieved_sources": metadatas
                }],
                "k": self.k
            }
            StudentSearchResults(**search_result)
            retrieved.append(search_result)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir,
                               f"dataset_{type_json}_public.json"), "w",
                  encoding="utf-8") as f:
            json.dump(retrieved, f, indent=4)
