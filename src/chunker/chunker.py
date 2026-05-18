from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from tqdm import tqdm
from .python_splitter import PythonSplitter


class Chunk:

    ignored_files: list = [
        "setup.py",
        ".toml",
        ".yaml",
        ".sh",
        ".gitignore",
        ".dockerignore",
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
            separators: list[str] | None = None,
            max_chunk_size: int = 3000,
            chunk_overlap: int = 500
    ):
        self.new_dir: str = "data/processed/chunks"
        os.makedirs(self.new_dir, exist_ok=True)
        self.type_file: str = type_file
        self.init_splitters(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        self.chunk_code: list = []
        self.chunk_text: list = []
        self.chunk_code_with_source: list = []
        self.chunk_text_with_source: list = []
        self.files: list = []

    def init_splitters(
            self,
            chunk_size: int,
            chunk_overlap: int,
            separators: list[str] = ["\n\n"]
    ) -> None:
        splitter_text: RecursiveCharacterTextSplitter = \
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
        splitter_code: PythonSplitter = PythonSplitter()
        self.splitters: dict = {
            "text": splitter_text,
            "code": splitter_code
        }

    def chunk(
            self,
            file_str: str
    ) -> list[str]:
        return self.splitters[self.type_file].split_text(file_str)

    def chunk_files(
            self
    ) -> None:
        for f in tqdm(self.files, desc="Chunking files"):
            current_pos: int = 0
            with open(f, "r", encoding="utf-8") as file:
                text = file.read()

            if f.endswith(".py"):
                self.type_file = "code"
            else:
                self.type_file = "text"

            chunks = self.chunk(text)
            for chunk in chunks:
                start_idx: int = text.find(chunk, current_pos)
                if start_idx == -1:
                    start_idx = text.find(chunk)
                end_idx: int = start_idx + len(chunk)
                current_pos = end_idx

                chunk_data = {
                    "chunk": chunk,
                    "file_path": f,
                    "first_character_index": start_idx,
                    "last_character_index": end_idx,
                }

                if f.endswith(".py"):
                    self.chunk_code.append(chunk)
                    self.chunk_code_with_source.append(chunk_data)
                else:
                    self.chunk_text.append(chunk)
                    self.chunk_text_with_source.append(chunk_data)

    def to_json(self) -> None:
        with open(os.path.join(self.new_dir, "chunk_text.json"), "w",
                  encoding="utf-8") as f:
            with tqdm(total=1, desc="Saving text chunks") as pbar:
                json.dump(self.chunk_text_with_source, f, indent=4)
                pbar.update(1)
        with open(os.path.join(self.new_dir, "chunk_code.json"), "w",
                  encoding="utf-8") as f:
            with tqdm(total=1, desc="Saving code chunks") as pbar:
                json.dump(self.chunk_code_with_source, f, indent=4)
                pbar.update(1)

    def get_files(
            self,
            directory: str
    ) -> None:
        all_dirs = os.listdir(directory)
        for direc in all_dirs:
            current_path = os.path.join(directory, direc)
            if os.path.abspath(current_path) == os.path.abspath(self.new_dir):
                continue
            if (direc.endswith(tuple(Chunk.ignored_files)) or
                    direc in Chunk.ignored_files):
                continue
            if os.path.isdir(current_path):
                self.get_files(current_path)
            else:
                self.files.append(current_path)
