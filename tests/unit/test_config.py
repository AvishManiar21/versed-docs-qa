from versed.config import Settings


def test_settings_reads_database_url_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://x:y@host/db")
    settings = Settings()
    assert settings.database_url == "postgresql+psycopg://x:y@host/db"


def test_settings_has_sane_defaults():
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.data_dir == "data"
