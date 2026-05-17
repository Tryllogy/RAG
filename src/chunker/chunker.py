from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from tqdm import tqdm


class Chunk:

    ignored_files: list = [
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
        self.chunk_code_with_source: list = []
        self.chunk_text_with_source: list = []
        self.files: list = []

    def chunk(
            self,
            file_str: str
    ) -> list[str]:
        if self.type_file == "text":
            return self.splitter.split_text(file_str)
        else:
            return self.splitter.from_language(file_str, language="python")

    def chunk_files(
            self
    ) -> None:
        for f in tqdm(self.files, desc="Chunking files"):
            current_pos: int = 0
            with open(f, "r") as file:
                text = file.read()
            chunks = self.chunk(text)
            for chunk in chunks:
                start_idx: int = text.find(chunk, current_pos)
                end_idx: int = start_idx + len(chunk)
                if f.endswith(".py"):
                    self.chunk_code.append(chunk)
                    self.chunk_code_with_source.append(
                        {"chunk": chunk,
                         "file_path": f,
                         "first_character_index": start_idx,
                         "last_character_index": end_idx}
                    )
                else:
                    self.chunk_text.append(chunk)
                    self.chunk_text_with_source.append(
                        {"chunk": chunk,
                         "file_path": f,
                         "first_character_index": start_idx,
                         "last_character_index": end_idx}
                    )

    def to_json(self) -> dict:
        with open(os.path.join(self.new_dir, "chunk_text.json"),
                  "w") as f:
            with tqdm(total=1, desc="Saving text chunks") as pbar:
                json.dump(self.chunk_text_with_source, f, indent=4)
                pbar.update(1)
        with open(os.path.join(self.new_dir, "chunk_code.json"),
                  "w") as f:
            with tqdm(total=1, desc="Saving code chunks") as pbar:
                json.dump(self.chunk_code_with_source, f, indent=4)
                pbar.update(1)

    def get_files(
            self,
            directory: str
    ) -> None:
        all_dirs = os.listdir(directory)
        for dir in all_dirs:
            if (dir.endswith(tuple(Chunk.ignored_files)) or
                    dir in Chunk.ignored_files):
                continue
            if os.path.isdir(os.path.join(directory, dir)):
                self.get_files(os.path.join(directory, dir))
            else:
                self.files.append(os.path.join(directory, dir))
