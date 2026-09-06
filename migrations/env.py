from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from versed.config import settings  # noqa: E402
from versed.db.models import Base  # noqa: E402

config.set_main_option("sqlalchemy.url", settings.database_url)
target_metadata = Base.metadata

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This project only ever runs migrations online, against a live
# DATABASE_URL (see settings.database_url above) — no offline/SQL-script
# generation mode is used, so that code path isn't implemented here.
connectable = engine_from_config(
    config.get_section(config.config_ini_section, {}),
    prefix="sqlalchemy.",
    poolclass=pool.NullPool,
)

with connectable.connect() as connection:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()
