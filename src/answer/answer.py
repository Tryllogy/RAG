from typing import Any
from ..retrieve import Retrieve
from llama_cpp import Llama
import json


class Answer:
    def __init__(
            self,
            question: str = "",
            k: int = 10
    ) -> None:
        """Initialize the Answer generator with question and
            retrieval settings.

        Args:
            question: The question to answer. Defaults to empty string.
            k: Number of documents to retrieve. Defaults to 10.
        """
        self.retriver: Retrieve = Retrieve()
        self.question: str = question
        self.k: int = k
        self.data_docs_path: str = "data/datasets/AnsweredQuestions/" \
            "dataset_docs_public.json"
        self.data_code_path: str = "data/datasets/AnsweredQuestions/" \
            "dataset_code_public.json"
        self.llm = Llama(
            model_path="./qwen-0.6b_Q8.gguf",
            n_ctx=7000,
            n_gpu_layers=-1,
            n_threads=8,
            temperature=0.7,
            max_tokens=500,
            verbose=False
        )
        self.prompt_message: str = """You are a question-answering assistant.
            Use only the information in CONTEXT.
            Do not end your phrase prematurely.
            Make ONLY 1 phrase.
            Finish with your phrase with '.'.
            If the answer is not present, say:
            'I don't know.'"""
        self.max_chunks: int = 3
        self.question_id: str = ""
        self.max_iter: int = 5
        self.retrieved_sources: list[dict[str, Any]] = []

    def answer(self) -> Any:
        """Generate an answer to the question using retrieved context.

        Uses the retriever to find relevant documents, constructs a prompt
        with the context, and queries the LLM for an answer.

        Returns:
            The generated answer as a string (removes thinking tags).
        """
        type_question: str = self.find_question(self.question)
        retrieved = self.retriver.retrieve(
            query=self.question, k=self.k, type_retriever=type_question)

        chunks: list[str] = [c.page_content for c in retrieved[
            :self.max_chunks]]
        self.retrieved_sources = [
            {
                "file_path": c.metadata.get("file_path"),
                "first_character_index":
                c.metadata.get("first_character_index"),
                "last_character_index": c.metadata.get("last_character_index")
            }
            for c in retrieved
        ]
        sources_text = "\n".join([
            f"[{i+1}] {c.metadata.get('file_path', 'unknown')} - Lines "
            f"{c.metadata.get('first_character_index', '?')}-"
            f"{c.metadata.get('last_character_index', '?')}"
            for i, c in enumerate(retrieved[:self.max_chunks])
        ])
        question: str = f"\n\nCONTEXT:\n" \
            f"{chr(10).join(chunks)}\n\nSOURCES:\n{sources_text}\n\n" \
            f"\n\n QUESTION: {self.question}\n\nANSWER:"
        result = self.llm(self.prompt_message + question)
        result = result["choices"][0]["text"].replace(
            "[THINKING]", "").replace("[/THINKING]", "").strip()
        i: int = 0
        while not result.endswith(".") and i < self.max_iter:
            result2 = self.llm(self.prompt_message + question + result)
            result = result + result2["choices"][0]["text"].replace(
                "[THINKING]", "").replace("[/THINKING]", "").strip()
            result = result.replace("\n", "").strip()
            i += 1
        return result

    def find_question(self, question: str) -> str:
        """Determine if a question is code-related or docs-related.

        Searches the code questions dataset to classify the question type.

        Args:
            question: The question text to classify.

        Returns:
            "code" if the question is in the code dataset, "docs" otherwise.
        """
        with open(self.data_code_path, "r") as f:
            content = json.load(f)
            content = content["rag_questions"]
        for c in content:
            if c["question"] == question:
                self.question_id = c["question_id"]
                return "code"
        with open(self.data_docs_path, "r") as f:
            content = json.load(f)
            content = content["rag_questions"]
        for c in content:
            if c["question"] == question:
                self.question_id = c["question_id"]
                return "docs"
        return "docs"
