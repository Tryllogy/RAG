import ast
from typing import Any


class PythonSplitter:

    def _split_content_if_needed(
        self,
        chunk: dict,
        max_length: int = 2000
    ) -> list[dict]:
        """Split a chunk if its content is too long."""
        content = chunk["content"]
        if len(content) <= max_length:
            return [chunk]

        sub_chunks = []
        num_parts = (len(content) + max_length - 1) // max_length

        for i in range(num_parts):
            start = i * max_length
            end = start + max_length
            sub_content = content[start:end]

            new_chunk = chunk.copy()
            new_chunk["content"] = sub_content
            new_chunk["name"] = f"{chunk['name']}_part_{i + 1}/{num_parts}"
            new_chunk["first_character_index"] = \
                chunk["first_character_index"] + start
            new_chunk["last_character_index"] = \
                chunk["first_character_index"] + end
            sub_chunks.append(new_chunk)

        return sub_chunks

    def split_text(self, source: str) -> list[dict]:
        """Parse Python source code and extract chunks by structure.

        Extracts module header, classes with their methods, and top-level
        functions as separate chunks with metadata.

        Args:
            source: The Python source code as a string.

        Returns:
            A list of dictionaries containing chunk content, metadata,
            and character offsets.
        """
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)

        line_offsets = []
        offset = 0
        for ln in lines:
            line_offsets.append(offset)
            offset += len(ln)

        chunks = []
        all_chunks = []

        module_header = self._extract_module_header(source, tree)
        if module_header:
            all_chunks.append({
                "content": module_header,
                "name": "__module__",
                "type": "module",
                "parent": None,
                "first_character_index": 0,
                "last_character_index": len(module_header),
            })

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_chunk = ast.get_source_segment(source, node)
                first, last = self._get_offsets(
                    node, line_offsets, source, class_chunk)
                all_chunks.append({
                    "content": class_chunk,
                    "name": node.name,
                    "type": "class",
                    "parent": None,
                    "first_character_index": first,
                    "last_character_index": last,
                })

                for child in ast.walk(node):
                    if isinstance(child, (ast.FunctionDef,
                                          ast.AsyncFunctionDef)):
                        method_src = ast.get_source_segment(source, child)
                        # Injecte le contexte parent
                        enriched = f"# Class: {node.name}\n{method_src}"
                        first, last = self._get_offsets(
                            child, line_offsets, source, method_src)
                        all_chunks.append({
                            "content": enriched,
                            "name": f"{node.name}.{child.name}",
                            "type": "method",
                            "parent": node.name,
                            "docstring": ast.get_docstring(child),
                            "first_character_index": first,
                            "last_character_index": last,
                        })

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not self._has_parent_class(tree, node):
                    func_src = ast.get_source_segment(source, node)
                    first, last = self._get_offsets(
                        node, line_offsets, source, func_src)
                    all_chunks.append({
                        "content": func_src,
                        "name": node.name,
                        "type": "function",
                        "parent": None,
                        "docstring": ast.get_docstring(node),
                        "first_character_index": first,
                        "last_character_index": last,
                    })

        for chunk in all_chunks:
            chunks.extend(self._split_content_if_needed(chunk))

        return chunks

    def _extract_module_header(
            self,
            source: str,
            tree: ast.Module
    ) -> str:
        """Extrait imports et variables top-level."""
        lines = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom,
                                 ast.Assign, ast.AnnAssign, ast.Expr)):
                segment = ast.get_source_segment(source, node)
                if segment:
                    lines.append(segment)
        return "\n".join(lines)

    def _has_parent_class(self, tree: ast.AST, target: ast.AST) -> bool:
        """Check if an AST node is a direct child of a class.

        Args:
            tree: The AST tree to search.
            target: The target node to check.

        Returns:
            True if the target is inside a ClassDef, False otherwise.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in ast.walk(node):
                    if child is target:
                        return True
        return False

    def _get_offsets(
        self,
        node: Any,
        line_offsets: list[int],
        source: str,
        chunk: str | None
    ) -> tuple[int, int]:
        """Calculate the character offsets for an AST node.

        Computes the start and end positions of a node in the source code
        using line offsets or fallback string search.

        Args:
            node: The AST node to get offsets for.
            line_offsets: List of character offsets for each line.
            source: The source code string.
            chunk: The source segment of the node (optional for fallback).

        Returns:
            A tuple of (start_offset, end_offset) in characters.
        """
        try:
            first = line_offsets[node.lineno - 1] + node.col_offset
            last = line_offsets[node.end_lineno - 1] + node.end_col_offset
        except Exception:
            first = source.find(chunk) if chunk else 0
            last = first + len(chunk) if chunk else 0
        return first, last
