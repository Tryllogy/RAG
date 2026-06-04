from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from tqdm import tqdm
from .python_splitter import PythonSplitter
from ..validators import MinimalSource


class Chunk:

    ignored_files: list = [
        "setup.py"
    ]

    authorized_files: list = [
        ".txt",
        ".py",
        ".md",
        ".rst"
    ]

    def __init__(
            self,
            type_file: str = "docs",
            separators: list[str] | None = None,
            max_chunk_size: int = 2000,
            chunk_overlap: int = 500
    ):
        """Initialize the document chunker.

        Args:
            type_file: Type of file to chunk ("docs" or "code"). Defaults
                to "docs".
            separators: List of separators for text splitting. Defaults to
                ["\\n\\n"].
            max_chunk_size: Maximum size of each chunk in characters.
                Defaults to 2000.
            chunk_overlap: Number of characters to overlap between chunks.
                Defaults to 500.
        """
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
            separators: list[str] | None = ["\n\n"]
    ) -> None:
        """Initialize text splitters for chunking.

        Creates and configures appropriate splitters based on file type
        (text or code).

        Args:
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            separators: List of separators to use for splitting. Defaults to
                ["\\n\\n"].
        """
        splitter_text: RecursiveCharacterTextSplitter = \
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
        splitter_code: PythonSplitter = PythonSplitter()
        self.splitters: dict = {
            "docs": splitter_text,
            "code": splitter_code
        }

    def chunk(self, file_str: str) -> Any:
        """Split text or code into chunks using the appropriate splitter.

        Args:
            file_str: The file content as a string.

        Returns:
            A list of chunks or chunk dictionaries.
        """
        return self.splitters[self.type_file].split_text(file_str)

    def chunk_files(self) -> None:
        """Process all files and split them into chunks.

        Iterates through the collected files, chunks them based on type,
        and stores both the raw chunks and metadata.
        """
        for f in tqdm(self.files, desc="Chunking files"):
            with open(f, "r", encoding="utf-8") as file:
                text = file.read()

            if len(text) == 0:
                continue
            if f.endswith(".py"):
                self.type_file = "code"
            else:
                self.type_file = "docs"

            chunks = self.chunk(text)

            for chunk_data in chunks:
                if isinstance(chunk_data, dict):
                    chunk = chunk_data["content"]
                    start_idx = chunk_data["first_character_index"]
                    end_idx = chunk_data["last_character_index"]
                    name = chunk_data.get("name", "unknown")
                    type_chunk = chunk_data.get("type", "unknown")
                    docstring = chunk_data.get("docstring", None)
                    parent = chunk_data.get("parent", None)
                else:
                    chunk = chunk_data
                    start_idx = text.find(chunk)
                    if start_idx == -1:
                        start_idx = 0
                    end_idx = start_idx + len(chunk)
                    name = "unknown"
                    type_chunk = "unknown"
                    docstring = None
                    parent = None

                chunk_info = {
                    "chunk": chunk,
                    "file_path": f,
                    "first_character_index": start_idx,
                    "last_character_index": end_idx,
                    "name": name,
                    "type": type_chunk,
                    "docstring": docstring,
                    "parent": parent
                }
                MinimalSource(**chunk_info)
                if f.endswith(".py"):
                    self.chunk_code.append(chunk)
                    self.chunk_code_with_source.append(chunk_info)
                else:
                    self.chunk_text.append(chunk)
                    self.chunk_text_with_source.append(chunk_info)

    def to_json(self) -> None:
        """Save chunked data to JSON files.

        Exports code and documentation chunks with their metadata to
        separate JSON files.
        """
        with open(os.path.join(self.new_dir, "chunk_docs.json"), "w",
                  encoding="utf-8") as f:
            with tqdm(total=1, desc="Saving docs chunks") as pbar:
                json.dump(self.chunk_text_with_source, f, indent=4)
                pbar.update(1)
        with open(os.path.join(self.new_dir, "chunk_code.json"), "w",
                  encoding="utf-8") as f:
            with tqdm(total=1, desc="Saving code chunks") as pbar:
                json.dump(self.chunk_code_with_source, f, indent=4)
                pbar.update(1)

    def get_files(self, directory: str) -> None:
        """Recursively collect all authorized files from a directory.

        Walks through the directory tree, ignoring specified files and
        collecting only authorized file types.

        Args:
            directory: The root directory to search.
        """
        all_dirs = os.listdir(directory)
        for direc in all_dirs:
            current_path = os.path.join(directory, direc)
            if os.path.abspath(current_path) == os.path.abspath(self.new_dir):
                continue
            elif (direc.endswith(tuple(Chunk.ignored_files)) or
                    direc in Chunk.ignored_files):
                continue
            elif os.path.isdir(current_path):
                self.get_files(current_path)
            elif (direc.endswith(tuple(Chunk.authorized_files)) or
                    direc in Chunk.authorized_files):
                self.files.append(current_path)
