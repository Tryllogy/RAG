import re


class ProcessCode:

    @staticmethod
    def _preprocess_code(text: str) -> list[str]:
        """Tokenize Python code in a BM25-friendly way.

        Split on non-alphanumeric boundaries and also break snake_case and
        camelCase identifiers so terms like `handle_connection_error` become
        searchable as `handle`, `connection`, and `error`.
        """
        tokens: list[str] = []
        for raw_token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
            parts = [part for part in raw_token.split("_") if part]
            for part in parts:
                sub_parts = re.findall(
                    r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+",
                    part,
                )
                if sub_parts:
                    tokens.extend(token.lower() for token in sub_parts)
                else:
                    tokens.append(part.lower())
        return tokens

    @staticmethod
    def _preprocess_text(text: str) -> list[str]:
        """Simple lowercased word tokenizer for prose chunks.

        Extracts all alphanumeric words and underscores from text,
        converting them to lowercase for BM25 indexing.

        Args:
            text: The input text to tokenize.

        Returns:
            A list of lowercased word tokens.
        """
        return re.findall(r"[A-Za-z0-9_]+", text.lower())
