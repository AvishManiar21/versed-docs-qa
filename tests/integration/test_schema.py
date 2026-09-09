import pytest
from sqlalchemy import inspect

from versed.db.session import engine

pytestmark = pytest.mark.integration


def test_tables_exist():
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {"chunk", "symbol", "symbol_event", "eval_question"} <= tables
