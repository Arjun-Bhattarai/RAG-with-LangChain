from typing import Any, List, Optional

from src.query_translation.multi_query import create_multi_query_generator


class QueryProcessor:
    """Expand a user query before retrieval using the existing Multi-Query generator."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        max_variations: int = 5,
        enable_multi_query: bool = True,
    ):
        self.max_variations = max(int(max_variations), 1)
        self.enable_multi_query = enable_multi_query
        self.generator = None
        if enable_multi_query:
            self.generator = create_multi_query_generator(llm=llm)

    def process(self, query: str) -> List[str]:
        original = (query or "").strip()
        if not original:
            return []

        queries = [original]
        if not self.enable_multi_query or self.generator is None:
            return queries

        try:
            variants = self._invoke_generator(original)
            normalized = self._normalize_variants(variants)
        except Exception:
            return queries

        if not normalized:
            return queries

        for text in normalized:
            if any(text.lower() == existing.lower() for existing in queries):
                continue
            queries.append(text)
            if len(queries) >= self.max_variations:
                break

        return queries

    def _invoke_generator(self, original: str) -> Any:
        payload = {"question": original}
        generator = self.generator
        type_error: Optional[TypeError] = None

        if callable(generator):
            try:
                return generator(payload)
            except TypeError as exc:
                type_error = exc
                try:
                    return generator(original)
                except TypeError as inner:
                    type_error = inner

        invoke = getattr(generator, "invoke", None)
        if callable(invoke):
            return invoke(payload)

        if type_error is not None:
            raise type_error
        raise RuntimeError("Query generator cannot be invoked.")

    @staticmethod
    def _normalize_variants(variants: Any) -> List[str]:
        if variants is None:
            return []

        if isinstance(variants, str):
            lines = variants.splitlines()
        elif isinstance(variants, (list, tuple)):
            lines = []
            for item in variants:
                if item is None:
                    continue
                text = str(item).strip()
                if not text:
                    continue
                lines.extend(text.splitlines())
        else:
            return []

        return [line.strip() for line in lines if line.strip()]
