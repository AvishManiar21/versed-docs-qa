import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^(```|~~~)")


@dataclass
class RawChunk:
    heading_path: list[str]
    content: str
    has_code: bool


def chunk_markdown(text: str) -> list[RawChunk]:
    """Split markdown/MDX into chunks on heading boundaries.

    Never splits inside a fenced code block. Each chunk's heading_path is
    the full stack of headings above it.
    """
    lines = text.splitlines()
    chunks: list[RawChunk] = []

    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append(
                RawChunk(
                    heading_path=[title for _, title in heading_stack],
                    content=content,
                    has_code="```" in content or "~~~" in content,
                )
            )
        current_lines.clear()

    for line in lines:
        stripped = line.strip()
        fence_match = _FENCE_RE.match(stripped)
        if fence_match:
            if not in_fence:
                in_fence = True
                fence_marker = fence_match.group(1)
            elif stripped.startswith(fence_marker):
                in_fence = False
            current_lines.append(line)
            continue

        if in_fence:
            current_lines.append(line)
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        current_lines.append(line)

    flush()
    return chunks
