from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_MODEL = "nomic-embed-text"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://versed:versed@localhost:5433/versed"
    data_dir: str = "data"


settings = Settings()
