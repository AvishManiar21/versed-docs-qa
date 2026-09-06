#!/usr/bin/env python3
"""Standalone symbol introspector.

STDLIB ONLY — no third-party imports. This script runs inside an isolated
per-version venv that has ONLY the target library installed, never this
project's own dependencies.

Usage: python introspect_worker.py <package_name>
Emits one JSON object per line (JSONL) to stdout.
"""

import importlib
import inspect
import json
import pkgutil
import re
import sys

_DEPRECATED_RE = re.compile(r"^\.\.\s*deprecated::\s*(\S+)(?:\s+(.*))?", re.DOTALL)
_ALTERNATIVE_RE = re.compile(r"[Uu]se(?: the)?\s+`{1,2}([^`]+)`{1,2}")


def parse_deprecation(doc):
    """Return (deprecated_since, alternative) parsed from a LangChain-style
    ``.. deprecated:: <since> <message>`` docstring header, or (None, None)
    if the docstring carries no deprecation notice.
    """
    if not doc:
        return None, None
    match = _DEPRECATED_RE.match(doc.strip())
    if not match:
        return None, None
    since_str = match.group(1)
    since = since_str.split("==")[-1] if "==" in since_str else since_str
    details = (match.group(2) or "").split("\n\n")[0].strip()
    alt_match = _ALTERNATIVE_RE.search(details)
    alternative = alt_match.group(1) if alt_match else None
    return since, alternative


def _safe_signature(obj):
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return ""


def _emit(qualified_name, kind, obj):
    doc = inspect.getdoc(obj)
    since, alternative = parse_deprecation(doc)
    print(
        json.dumps(
            {
                "qualified_name": qualified_name,
                "kind": kind,
                "signature": _safe_signature(obj),
                "docstring": (doc or "")[:2000],
                "deprecated_since": since,
                "alternative": alternative,
            }
        )
    )


def walk_public_symbols(package_name):
    package = importlib.import_module(package_name)
    seen_modules = set()

    def visit_module(module):
        if module.__name__ in seen_modules:
            return
        seen_modules.add(module.__name__)

        for member_name, member in inspect.getmembers(module):
            if member_name.startswith("_"):
                continue
            if (
                inspect.isfunction(member)
                and getattr(member, "__module__", None) == module.__name__
            ):
                _emit(f"{module.__name__}.{member_name}", "function", member)
            elif inspect.isclass(member) and getattr(member, "__module__", None) == module.__name__:
                _emit(f"{module.__name__}.{member_name}", "class", member)
                for method_name, method in inspect.getmembers(member, predicate=inspect.isfunction):
                    if method_name.startswith("_"):
                        continue
                    _emit(f"{module.__name__}.{member_name}.{method_name}", "method", method)

    if hasattr(package, "__path__"):
        prefix = package.__name__ + "."
        for _, name, _ in pkgutil.walk_packages(package.__path__, prefix=prefix):
            try:
                module = importlib.import_module(name)
            except Exception:
                continue
            visit_module(module)
    visit_module(package)


if __name__ == "__main__":
    walk_public_symbols(sys.argv[1])
