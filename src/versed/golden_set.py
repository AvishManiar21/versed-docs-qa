import uuid
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Category = Literal[
    "version_explicit", "version_implicit", "migration", "removed_api", "unanswerable"
]


class GoldenQuestion(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    question: str
    category: Category
    target_version: str | None = None
    expected_answer: str | None = None
    expected_symbols: list[str] = Field(default_factory=list)
    should_abstain: bool = False
    reviewed: bool = False
    reviewer_note: str | None = None

    @model_validator(mode="after")
    def _abstain_has_no_answer(self) -> "GoldenQuestion":
        if self.should_abstain and self.expected_answer is not None:
            raise ValueError("should_abstain rows must not carry an expected_answer")
        return self


def write_golden_set(questions: list[GoldenQuestion], path: Path) -> None:
    payload = [q.model_dump(mode="json") for q in questions]
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def read_golden_set(path: Path) -> list[GoldenQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [GoldenQuestion.model_validate(row) for row in raw]
