import ast
from typing import Any


class PythonSplitter:

    def split_text(
            self,
            source: str
    ) -> list[dict]:
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)

        line_offsets = []
        offset = 0
        for ln in lines:
            line_offsets.append(offset)
            offset += len(ln)

        chunks = []

        module_header = self._extract_module_header(source, tree)
        if module_header:
            chunks.append({
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
                chunks.append({
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
                        chunks.append({
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
                    chunks.append({
                        "content": func_src,
                        "name": node.name,
                        "type": "function",
                        "parent": None,
                        "docstring": ast.get_docstring(node),
                        "first_character_index": first,
                        "last_character_index": last,
                    })

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

    def _has_parent_class(
            self,
            tree: ast.AST,
            target: ast.AST
    ) -> bool:
        """Vérifie si un nœud est dans une ClassDef."""
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
        try:
            first = line_offsets[node.lineno - 1] + node.col_offset
            last = line_offsets[node.end_lineno - 1] + node.end_col_offset
        except Exception:
            first = source.find(chunk) if chunk else 0
            last = first + len(chunk) if chunk else 0
        return first, last
