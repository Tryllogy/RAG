import fire
from langchain_ollama import OllamaLLM
from src import indexer, retriever


def main() -> None:
    llm = OllamaLLM(model="qwen3")
    # fire.Fire(indexer.Index(llm))
    fire.Fire(retriever.Retrieve())


if __name__ == "__main__":
    main()
