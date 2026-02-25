from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENVIRONMENT: str = "dev"

    DATABASE_URL: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    COOKIE_DOMAIN: str | None = None
    COOKIE_SAMESITE: str = "Lax"
    COOKIE_SECURE: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()