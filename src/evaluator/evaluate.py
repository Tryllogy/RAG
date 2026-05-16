import json
from pathlib import Path


class Evaluate:
    def __init__(self):
        self.sources: list | None = None
        self.overlap_threshold: float = 0.05

    def _calculate_overlap(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        max_len = max(len(words1), len(words2))

        return overlap / max_len

    def evaluate(self, results, k: int = None):
        if not self.sources:
            return 0.0

        k = k if k else len(results)
        retrieved_at_k = results[:k]

        retrieved_chunks = []
        for item in retrieved_at_k:
            if hasattr(item, 'page_content') and \
               hasattr(item, 'metadata'):
                content = item.page_content
                source = item.metadata.get("source")
                retrieved_chunks.append({"content": content,
                                        "source": source})
            elif isinstance(item, dict):
                retrieved_chunks.append(item)

        correct_sources = []
        for source_info in self.sources:
            file_path = source_info.get("file_path")
            if file_path and Path(file_path).exists():
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                    correct_sources.append({"file_path": file_path,
                                           "content": content})
                except Exception:
                    pass

        if not correct_sources:
            return 0.0

        found_sources = set()
        for retrieved in retrieved_chunks:
            retrieved_content = retrieved.get("content", "")

            for idx, correct in enumerate(correct_sources):
                correct_content = correct["content"]
                overlap = self._calculate_overlap(
                    retrieved_content,
                    correct_content
                )

                if overlap >= self.overlap_threshold:
                    found_sources.add(idx)

        recall_k = len(found_sources) / len(correct_sources)
        return recall_k

    def get_source_question(self, question: str) -> list:
        code_path = ("data/datasets_public/"
                     "AnsweredQuestions/dataset_code_public.json")
        docs_path = ("data/datasets_public/"
                     "AnsweredQuestions/dataset_docs_public.json")

        with open(code_path, "r") as f:
            data = json.load(f)
            for item in data["rag_questions"]:
                if item["question"] == question:
                    self.sources = item.get("sources")
                    return item.get("sources")

        with open(docs_path, "r") as f:
            data = json.load(f)
            for item in data["rag_questions"]:
                if item["question"] == question:
                    self.sources = item.get("sources")
                    return item.get("sources")

        return None
