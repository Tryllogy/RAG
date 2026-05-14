from langchain_ollama import OllamaLLM
from ..index import Index
import os


class FireLaunch:
    def __init__(self, max_chunk_size: int = 3000):
        self.llm: OllamaLLM = OllamaLLM(model="qwen3")
        self.max_chunk_size: int = max_chunk_size

    def index(self):
        if os.path.exists("data/processed/chunks") and \
            os.path.exists("data/processed/bm25_index"):
            print("Files already chunked")
            return self
        index_instance = Index(llm=self.llm,
                               max_chunk_size=self.max_chunk_size)
        index_instance.index()
        return self

    def answer(self):
        print("answer")
        return self

    def search_dataset(self):
        pass

    def answer_dataset(self):
        pass

    def __str__(self):
        return "Done"
