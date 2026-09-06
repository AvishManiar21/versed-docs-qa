import re

import tiktoken

_INLINE_CODE_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`")
_encoding = tiktoken.get_encoding("cl100k_base")


def extract_symbols_mentioned(content: str) -> list[str]:
    return sorted(set(_INLINE_CODE_RE.findall(content)))


def count_tokens(content: str) -> int:
    return len(_encoding.encode(content))
