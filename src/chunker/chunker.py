from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json


class Chunk:

    IGNORED_FILES = [
        "setup.py",
        ".toml",
        ".yaml",
        ".sh",
        ".gitignore",
        ".dockerignore"
        ".clang-format",
        ".shellcheckrc",
        ".yapfignore",
        "DCO",
        ".jinja",
        ".png",
        ".jpg",
        ".patch",
        ".ico"
    ]

    def __init__(
        self,
            type_file: str = "text",
            separators: list[str] = ["\n\n"],
            max_chunk_size: int = 3000
    ):
        self.new_dir: str = "data/processed/chunks"
        os.makedirs(self.new_dir, exist_ok=True)
        self.type_file: str = type_file
        self.splitter: RecursiveCharacterTextSplitter = \
            RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                separators=separators
            )
        self.chunk_code: list = []
        self.chunk_text: list = []

    def chunk(
            self,
            file_str: str
    ) -> list[str]:
        if self.type_file == "text":
            return self.splitter.split_text(file_str)
        else:
            return self.splitter.from_language(file_str, language="python")

    def chunk_files(
            self,
            directory: str
    ) -> None:
        all_dirs = os.listdir(directory)
        for dir in all_dirs:
            if (dir.endswith(tuple(self.IGNORED_FILES)) or
                    dir in self.IGNORED_FILES):
                continue
            if os.path.isdir(os.path.join(directory, dir)):
                self.chunk_files(os.path.join(directory, dir))
            else:
                with open(os.path.join(directory, dir), "r") as f:
                    text = f.read()
                chunks = self.chunk(text)
                for chunk in chunks:
                    if dir.endswith(".py"):
                        self.chunk_code.append(chunk)
                    else:
                        self.chunk_text.append(chunk)

    def to_json(self) -> dict:
        chunk_text: list = []
        chunk_code: list = []
        for chunk in self.chunk_text:
            chunk_text.append({"chunk": chunk})
        for chunk in self.chunk_code:
            chunk_code.append({"chunk": chunk})

        with open(os.path.join(self.new_dir, "chunk_text.json"), "w") as f:
            json.dump(chunk_text, f, indent=4)
        with open(os.path.join(self.new_dir, "chunk_code.json"), "w") as f:
            json.dump(chunk_code, f, indent=4)
