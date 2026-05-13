import fire
from langchain_ollama import OllamaLLM
from src import indexer


def test(x="salut"):
    return f"{x} Feurlogy!"


def main() -> None:
    llm = OllamaLLM(model="qwen3")
    fire.Fire(indexer.Index(llm))


if __name__ == "__main__":
    main()
