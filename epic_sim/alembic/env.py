"""Alembic migration environment."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so epic_sim is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from epic_sim.app.config import settings
from epic_sim.app.models import Base  # noqa: F401 — registers all models

config = context.config
# The ini file hard-codes a localhost URL; prefer the application setting so that
# EPIC_SIM_DATABASE_URL_SYNC (e.g. inside a container) targets the right database.
config.set_main_option("sqlalchemy.url", settings.database_url_sync)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
