from ..retrieve import Retrieve
import json
from tqdm import tqdm


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
            k: int = 15
    ) -> tuple[dict, dict]:
        recall_k_code: dict = {
            "1": 0.0,
            "3": 0.0,
            "5": 0.0,
            "10": 0.0
        }
        recall_k_text: dict = {
            "1": 0.0,
            "3": 0.0,
            "5": 0.0,
            "10": 0.0
        }
        n_questions_text: int = 0
        n_questions_code: int = 0

        with open(self.code_source, "r") as f:
            content = json.load(f)
            content = content["rag_questions"]
        with tqdm(content, desc="Evaluating code questions") as pbar:
            for c in content:
                n_questions_code += 1
                for key, v in recall_k_code.items():
                    found = self.get_found(
                        query=c["question"], k=int(key), type_retriever="code")
                    recall_k_code[key] += found
                pbar.update(1)

        with open(self.text_source, "r") as f:
            content = json.load(f)
            content = content["rag_questions"]
        with tqdm(content, desc="Evaluating text questions") as pbar:
            for c in content:
                n_questions_text += 1
                for key, v in recall_k_text.items():
                    found = self.get_found(
                        query=c["question"], k=int(key), type_retriever="text")
                    recall_k_text[key] += found
                pbar.update(1)

        for key, v in recall_k_code.items():
            recall_k_code[key] = round(v / n_questions_code, 3)
        for key, v in recall_k_text.items():
            recall_k_text[key] = round(v / n_questions_text, 3)
        return recall_k_code, recall_k_text

    def get_found(
            self,
            query: str,
            type_retriever: str,
            k: int = 15,
    ) -> float:
        found: int = 0
        results = self.retriever.retrieve(
            query, k=k, type_retriever=type_retriever)
        origin_sources = self.get_sources_question(query)

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
            query: str
    ) -> list | None:
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
