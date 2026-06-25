from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    APP_NAME: str

    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    REDIS_URL: str
    RABBITMQ_URL: str

    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()