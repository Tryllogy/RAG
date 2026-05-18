import ast


class PythonSplitter:

    def split_text(
            self,
            text: str
    ) -> list[str]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [text]

        chunks: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef,
                                 ast.ClassDef,
                                 ast.AsyncFunctionDef)):
                start_line = node.lineno - 1
                end_line = node.end_lineno
                chunk = "\n".join(text.splitlines()[start_line:end_line])
                chunks.append(chunk)
        return chunks
