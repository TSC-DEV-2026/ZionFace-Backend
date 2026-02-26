from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    ENVIRONMENT: str = Field(default="dev")
    DB_HOST: str
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DATABASE_URL: Optional[str] = None

    JWT_SECRET_KEY: str = Field(alias="SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", alias="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30, alias="REFRESH_TOKEN_EXPIRE_DAYS")

    COOKIE_DOMAIN: str = ""
    COOKIE_SAMESITE: str = "Lax"
    COOKIE_SECURE: bool = False

    FACE_MODEL_PATH: str = "models/arcface.onnx"
    FACE_INPUT_SIZE: int = 112
    FACE_VERIFY_THRESHOLD: float = 0.35

    def build_database_url(self) -> str:
        if self.DATABASE_URL and self.DATABASE_URL.strip():
            return self.DATABASE_URL.strip()
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


settings = Settings()