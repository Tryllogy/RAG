from ..retrieve import Retrieve
import json


class Evaluate:
    def __init__(self):
        self.sources: list | None = None
        self.overlap_threshold: float = 0.05
        self.retriever = Retrieve()
        self.code_source = "data/datasets_public/" \
            "AnsweredQuestions/dataset_code_public.json"
        self.text_source = "data/datasets_public/" \
            "AnsweredQuestions/dataset_docs_public.json"

    def evaluate(
            self,
            query: str,
            k: int = 15
    ) -> tuple[dict, dict]:
        recall_k_code: dict = {
            "1": 0,
            "3": 0,
            "5": 0,
            "10": 0
        }
        recall_k_text: dict = {
            "1": 0,
            "3": 0,
            "5": 0,
            "10": 0
        }
        n_questions_text: int = 0
        n_questions_code: int = 0

        with open(self.code_source, "r") as f:
            content = json.load(f)
            content = content["rag_questions"]
        for c in content:
            n_questions_code += 1
            for k, v in recall_k_code.items():
                recall_k_code[k] += self.get_found(
                    query=c["question"], k=int(k))

        with open(self.text_source, "r") as f:
            content = json.load(f)
            content = content["rag_questions"]
        for c in content:
            n_questions_text += 1
            for k, v in recall_k_text.items():
                recall_k_text[k] += self.get_found(
                    query=c["question"], k=int(k))

        return recall_k_code, recall_k_text

    def get_found(
            self,
            query: str,
            k: int = 15
    ) -> int:
        found: int = 0
        results = self.retriever.retrieve(query, k=k)
        origin_sources = self.get_sources_question(query)

        if origin_sources is None:
            return 0

        for r in results:
            metadata = r.metadata
            for source in origin_sources:
                if not source["file_path"] == metadata["file_path"]:
                    continue

                overlap = self.get_overlap(source, metadata)
                if overlap >= self.overlap_threshold:
                    found += 1

        if found == 0:
            return 0

        return (found / len(origin_sources))

    def get_overlap(
            self,
            source: str,
            metadata: dict
    ) -> int:
        range_s: list = [i for i in
                         range(source["first_character_index"],
                               source["last_character_index"])]
        range_m: list = [i for i in
                         range(metadata["first_character_index"],
                               metadata["last_character_index"])]
        in_common: int = len(set(range_s) & set(range_m))

        if in_common == 0:
            return 0

        len_s: int = source["last_character_index"] - \
            source["first_character_index"]

        return min(int(in_common / len_s), 100)

    def get_sources_question(
            self,
            query: str
    ) -> str:
        with open(self.code_source, "r") as f:
            content = json.load(f)
        for c in content["rag_questions"]:
            if c["question"] == query:
                return c["sources"]

        with open(self.text_source, "r") as f:
            content = json.load(f)
        for c in content["rag_questions"]:
            if c["question"] == query:
                return c["sources"]
        return None
