from ..retrieve import Retrieve
import json
from tqdm import tqdm


class Evaluate:
    def __init__(
            self,
            student_answer_path: str,
            dataset_path: str
    ):
        self.overlap_threshold: float = 0.05
        self.retriever = Retrieve()
        self.student_answer_path: str = student_answer_path
        self.dataset_path: str = dataset_path

    def evaluate(
            self,
            k: int = 15
    ) -> tuple[dict, dict]:
        recall_k: dict = {
            "1": 0.0,
            "3": 0.0,
            "5": 0.0,
            "10": 0.0
        }
        n_questions: int = 0

        type_file: str = self.student_answer_path.split("/")[-1]
        type_file = type_file.split("_")[1]
        with open(self.student_answer_path, "r") as f:
            content = json.load(f)
        with tqdm(content, desc="Evaluating questions") as pbar:
            for c in content:
                c = c["search_results"][0]
                n_questions += 1
                for key, v in recall_k.items():
                    found = self.get_found(
                        query=c["question"], question_id=c["question_id"],
                        k=int(key), type_retriever=type_file)
                    recall_k[key] += found
                pbar.update(1)

        for key, v in recall_k.items():
            recall_k[key] = round(v / n_questions, 3)
        return recall_k

    def get_found(
            self,
            query: str,
            question_id: str,
            type_retriever: str,
            k: int = 15,
    ) -> float:
        found: int = 0
        results = self.retriever.retrieve(
            query, k=k, type_retriever=type_retriever)
        origin_sources = self.get_sources_question(question_id)

        if origin_sources is None:
            return 0.0

        for source in origin_sources:
            for r in results:
                metadata = r.metadata
                if not source["file_path"] == metadata["file_path"]:
                    continue

                overlap = self.get_overlap(source, metadata)
                if overlap >= self.overlap_threshold:
                    found += 1
                    break

        return round(found / len(origin_sources), 3)

    def get_overlap(
            self,
            source: dict,
            metadata: dict
    ) -> float:
        source_start = source["first_character_index"]
        source_end = source["last_character_index"]
        metadata_start = metadata["first_character_index"]
        metadata_end = metadata["last_character_index"]

        len_s = source_end - source_start
        if len_s == 0:
            return 0.0

        overlap = max(0, min(source_end, metadata_end) -
                      max(source_start, metadata_start))
        return float(overlap / len_s)

    def get_sources_question(
            self,
            question_id: str
    ) -> list | None:
        with open("data/datasets/AnsweredQuestions/dataset_code_public.json",
                  "r") as f:
            content = json.load(f)
        for c in content["rag_questions"]:
            if c["question_id"] == question_id:
                return c["sources"]

        with open("data/datasets/AnsweredQuestions/dataset_docs_public.json",
                  "r") as f:
            content = json.load(f)
        for c in content["rag_questions"]:
            if c["question_id"] == question_id:
                return c["sources"]
        return None
